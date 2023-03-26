from abc import ABC, abstractmethod
from typing import List
from torch.nn import functional as F
from torch import nn
import torch
import scipy
import numpy as np
from motionblur.motionblur import Kernel

class Op(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # generic forward
        pass

    def backproject(self, data, **kwargs):
        # generic backproject
        pass

class OpCompose(Op):
    def __init__(self, op_list: List[Op]):
        self.op_list = op_list

    def forward(self, data):
        for op in self.op_list:
            data = op.forward(data)
        return data
    
    def backproject(self, data, **kwargs):
        for op in self.op_list:
            data = op.backproject(data)
        return data

class LinearOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

class NonLinearOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

class NoiseOp(Op):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate b
        pass



class SuperResolutionOp(LinearOp):
    def __init__(self, scale_factor, mode="bilinear"):
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, data):
        return F.interpolate(
            data,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False,
            antialias=True,
        )
    
    def backproject(self, data, **kwargs):
        return F.interpolate(
            data,
            scale_factor=1 / self.scale_factor,
            mode=self.mode,
            align_corners=False,
            antialias=True,
        )


class Blurkernel(nn.Module):
    def __init__(self, blur_type="gaussian", kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(
                3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3
            ),
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(
                size=(self.kernel_size, self.kernel_size), intensity=self.std
            ).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class MotionBlurOp(LinearOp):
    def __init__(self, kernel_size, intensity, device) -> None:
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.device = device
        self.conv = Blurkernel(
            blur_type="motion",
            kernel_size=self.kernel_size,
            std=self.intensity,
            device=self.device,
        )

        self.kernel = Kernel(
            size=(self.kernel_size, self.kernel_size), intensity=self.intensity
        )
        kernel = torch.tensor(self.Kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def forward(self, data):
        return self.conv(data)
    
    def backproject(self, data, **kwargs):
        return data


class GaussianBlurOp(LinearOp):
    def __init__(self, kernel_size, std, device) -> None:
        self.kernel_size = kernel_size
        self.std = std
        self.device = device

        self.conv = Blurkernel(
            blur_type="gaussian",
            kernel_size=self.kernel_size,
            std=self.std,
            device=device,
        )

        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data):
        return self.conv(data)
    
    def backproject(self, data, **kwargs):
        return data


class CenterBoxInpaintingOP(LinearOp):
    def __init__(self, im_size, box_size):
        self.box_size = box_size
        self.im_size = im_size
        self.mask = torch.ones((1, 1, self.im_size, self.im_size))
        self.mask[
            :,
            :,
            self.im_size // 2
            - self.box_size // 2 : self.im_size // 2
            + self.box_size // 2,
            self.im_size // 2
            - self.box_size // 2 : self.im_size // 2
            + self.box_size // 2,
        ] = 0

    def forward(self, data):
        return data * self.mask
    
    def backproject(self, data, **kwargs):
        return data


class NonlinearBlurOp(NonLinearOp):
    def __init__(self, opt_yaml_path, pretrained_model_path, device) -> None:
        super().__init__()
        self.opt_yaml_path = opt_yaml_path
        self.pretrained_model_path = pretrained_model_path
        self.device = device

        self.model = self.load_nonlinear_blur_model(
            self.opt_yaml_path, self.pretrained_model_path, self.device
        )
    
    @staticmethod
    def load_nonlinear_blur_model(opt_yaml_path, pretrained_model_path, device):
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard
        import yaml

        with open(opt_yaml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
        model = KernelWizard(opt)
        model.eval()
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        return model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn((1, 512, 2, 2), device=self.device) * 1.2
        data = (data + 1.0) / 2.0
        blurred_data = self.model.adaptKernel(data, random_kernel)
        blurred_data = ((blurred_data - 0.5) * 2.0).clamp(-1.0, 1.0)
        return blurred_data

class GaussianNoiseOp(NoiseOp):
    def __init__(self, std, device) -> None:
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, data, **kwargs):
        noise = torch.randn_like(data) * self.std
        return data + noise
    
class PoissonNoiseOp(NoiseOp):
    def __init__(self, rate, device) -> None:
        super().__init__()
        self.rate = rate
        self.device = device
        self.poisson = torch.distributions.poisson.Poisson(rate)

    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0
        data = data.clamp(0.0, 1.0)
        noise = self.poisson.sample(data.shape).to(self.device)
        noise = (noise / 255.0) / self.rate
        data = data * noise
        data = data * 2.0 - 1.0
        data = data.clamp(-1.0, 1.0)
        return data