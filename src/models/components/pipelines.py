from typing import List, Optional, Tuple, Union
import PIL
from diffusers import (
    VQModel,
    UNet2DModel,
    ImagePipelineOutput,
    DDPMScheduler,
    DiffusionPipeline,
)
from diffusers.utils import randn_tensor
import pyrootutils
import torch

from .guidence_modules import PixelGuidenceModule


class PixelReconstructionPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        condition_module: PixelGuidenceModule,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            unet=unet,
            condition_module=condition_module,
            scheduler=scheduler,
        )

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        noise_shape = (
            measurement.shape,
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
        )
        noise = randn_tensor(
            noise_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        losses = []
        with torch.enable_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                noise.requires_grad = True
                model_output = self.unet(noise, t).sample
                model_output = self.scheduler.scale_model_input(model_output, t)
                x_samples = self.scheduler.step(
                    model_output,
                    t,
                    noise,
                )

                x_prev = x_samples.prev_sample
                x0_hat = x_samples.pred_original_sample

                x_t, diff_val, l2_dist = self.condition_module(
                    x_prev=x_prev, x_t=noise, x_0_hat=x0_hat, measurement=measurement
                )

                losses.append(l2_dist)
                noise = x_t.detach()

        image = noise
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)


class LatentReconstructionPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        vae: VQModel,
        condition_module: PixelGuidenceModule,
        scheduler: DDPMScheduler,
        latents_height,
        latents_width,
    ) -> None:
        super().__init__()
        condition_module = condition_module(vae=vae)
        self.register_modules(
            unet=unet,
            vae=vae,
            condition_module=condition_module,
            scheduler=scheduler,
        )

        self.latents_height = latents_height
        self.latents_width = latents_width

    def __call__(
        self,
        measurement,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        latent_shape = (
            measurement.shape,
            self.unet.in_channels,
            self.latents_height,
            self.latents_width,
        )
        latents = randn_tensor(
            latent_shape,
            generator=generator,
            device=measurement.device,
            dtype=self.unet.dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        losses = []
        with torch.enable_grad():
            for t in self.progress_bar(self.scheduler.timesteps):
                latents.requires_grad = True
                model_output = self.unet(latents, t).sample
                model_output = self.scheduler.scale_model_input(model_output, t)
                l_samples = self.scheduler.step(
                    model_output,
                    t,
                    latents,
                )

                l_prev = l_samples.prev_sample
                l0_hat = l_samples.pred_original_sample

                l_t, diff_val, l2_dist = self.condition_module(
                    l_prev=l_prev, l_t=latents, l_0_hat=l0_hat, measurement=measurement
                )

                losses.append(l2_dist)
                latents = l_t.detach()

        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,), losses

        return ImagePipelineOutput(images=image)
