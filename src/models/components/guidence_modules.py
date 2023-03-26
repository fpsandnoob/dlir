from abc import ABC, abstractmethod
import torch
from typing import Tuple
from diffusers import VQModel

from .measurement_ops import NonLinearOp, LinearOp, OpCompose, NoiseOp


class Optimizer(ABC):
    @abstractmethod
    def __call__(self, x, grad, scale):
        pass


class GenericGuidenceModule(ABC):
    def __init__(
        self,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
    ) -> None:
        self.degraed_op = degraed_op
        self.noise_op = noise_op
        super().__init__()

    @abstractmethod
    def grad_and_diff(self, data, **kwargs):
        # generic forward
        pass

    @abstractmethod
    def conditioning(self, data, measurements, **kwargs):
        pass


class SGDOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x, grad, scale):
        return x - scale * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, scale, momentum=0.9) -> None:
        super().__init__()
        self.momentum = momentum
        self.m = None

    def __call__(self, x, grad, scale):
        if self.m is None:
            self.m = grad
        self.m = self.momentum * grad + (1 - self.momentum) * self.m
        return x - scale * self.m


class AdamOptimizer(Optimizer):
    def __init__(self, betas=(0.9, 0.999)) -> None:
        super().__init__()
        self.betas = betas
        self.m = None
        self.v = None

    def __call__(self, x, grad, scale):
        if self.m is None:
            self.m = torch.zeros_like(grad).type_as(grad)
        if self.v is None:
            self.v = torch.zeros_like(grad).type_as(grad)
        self.m = (1 - self.betas[0]) * grad + self.betas[0] * self.m
        self.v = (1 - self.betas[1]) * grad**2 + self.betas[1] * self.v

        self.m_hat = self.m / (1 - self.betas[0])
        self.v_hat = self.v / (1 - self.betas[1])

        return x - scale * self.m_hat / (torch.sqrt(self.v_hat) + 1e-8)


class PixelGuidenceModule(GenericGuidenceModule):
    def __init__(
        self,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degraed_op, noise_op)
        self.diff_module = diff_module
        self.l2_dist = torch.nn.MSELoss()

    def grad_and_diff(self, x_prev, x_0_hat, measurements, **kwargs):
        # calculate grad and diff
        deg_x_0_hat = self.degraed_op.forward(x_0_hat)
        diff_val = self.diff_module(deg_x_0_hat, measurements)
        l2_dist = self.l2_dist(deg_x_0_hat, measurements)

        norm = torch.linalg.norm(diff_val)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        return norm_grad, diff_val, l2_dist


class LatentGuidenceModule(GenericGuidenceModule):
    def __init__(
        self,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degraed_op, noise_op)
        self.diff_module = diff_module
        self.vae = vae

    def grad_and_diff(self, l_prev, l_0_hat, measurements, **kwargs):
        deg_x_0_hat = self.degraed_op.forward(self.vae.decode(l_0_hat).sample)
        diff_val = self.diff_module(deg_x_0_hat, measurements)
        l2_dist = self.l2_dist(deg_x_0_hat, measurements)

        norm = torch.linalg.norm(diff_val)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=l_prev)[0]

        return norm_grad, diff_val, l2_dist


class PixelManifoldConstraintGradient(PixelGuidenceModule):
    def __init__(
        self,
        scale,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degraed_op, noise_op, diff_module)
        self.scale = scale

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # calculate grad and diff
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t = x_t - self.scale * norm_grad

        # X + A^T(Y - AX)
        x_t = x_t + self.degraed_op.backproject(
            measurement - self.degraed_op.forward(x_t)
        )
        return x_t, diff_val, l2_dist


class LatentManifoldConstraintGradient(LatentGuidenceModule):
    def __init__(
        self,
        scale,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degraed_op, noise_op, diff_module, vae)
        self.scale = scale

    def conditioning(self, l_prev, l_t, l_0_hat, measurement, **kwargs):
        # calculate grad and diff
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            l_prev=l_prev, l_0_hat=l_0_hat, measurement=measurement, **kwargs
        )
        l_t = l_t - self.scale * norm_grad

        # E(D(L) + A^T(Y - AD(L)))
        decode_l_t = self.vae.decode(l_t).sample
        l_t = self.vae.encode(
            decode_l_t
            + self.degraed_op.backproject(
                measurement - self.degraed_op.forward(decode_l_t)
            )
        ).latents
        return l_t, diff_val, l2_dist


class PixelDeepLatentIterativeReconstruct(PixelGuidenceModule):
    def __init__(
        self,
        scale,
        optimizer,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
    ) -> None:
        super().__init__(degraed_op, noise_op, diff_module)
        self.scale = scale
        self.optimizer = optimizer

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # calculate grad and diff
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs
        )
        x_t = self.optimizer(x_t, norm_grad, self.scale)

        return x_t, diff_val, l2_dist


class LatentDeepLatentIterativeReconstruct(LatentGuidenceModule):
    def __init__(
        self,
        scale,
        optimizer,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        diff_module: torch.nn.Module,
        vae: VQModel,
    ) -> None:
        super().__init__(degraed_op, noise_op, diff_module, vae)
        self.scale = scale
        self.optimizer = optimizer

    def conditioning(self, l_prev, l_t, l_0_hat, measurement, **kwargs):
        # calculate grad and diff
        norm_grad, diff_val, l2_dist = self.grad_and_diff(
            l_prev=l_prev, l_0_hat=l_0_hat, measurement=measurement, **kwargs
        )
        l_t = self.optimizer(l_t, norm_grad, self.scale)

        return l_t, diff_val, l2_dist


class PixelPosteriorSampling(PixelDeepLatentIterativeReconstruct):
    def __init__(
        self,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
    ) -> None:
        super().__init__(1, SGDOptimizer(), degraed_op, noise_op, torch.nn.MSELoss())

class LatentPosteriorSampling(LatentDeepLatentIterativeReconstruct):
    def __init__(
        self,
        degraed_op: Tuple[OpCompose, LinearOp, NonLinearOp],
        noise_op: Tuple[OpCompose, NoiseOp],
        vae: VQModel,
    ) -> None:
        super().__init__(1, SGDOptimizer(), degraed_op, noise_op, torch.nn.MSELoss(), vae)
