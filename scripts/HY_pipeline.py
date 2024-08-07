# Copyright 2024 HunyuanDiT Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from transformers import CLIPImageProcessor

from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, HunyuanDiT2DModel
from diffusers.loaders import SD3LoraLoaderMixin

from diffusers.models import HunyuanDiT2DControlNetModel#, HunyuanDiT2DMultiControlNetModel

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


STANDARD_RATIO = np.array(
    [
        1.0,  # 1:1
        4.0 / 3.0,  # 4:3
        3.0 / 4.0,  # 3:4
        16.0 / 9.0,  # 16:9
        9.0 / 16.0,  # 9:16
    ]
)
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],  # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],  # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],  # 3:4
    [(1280, 768)],  # 16:9
    [(768, 1280)],  # 9:16
]
STANDARD_AREA = [np.array([w * h for w, h in shapes]) for shapes in STANDARD_SHAPE]
SUPPORTED_SHAPE = [
    (1024, 1024),
    (1280, 1280),  # 1:1
    (1024, 768),
    (1152, 864),
    (1280, 960),  # 4:3
    (768, 1024),
    (864, 1152),
    (960, 1280),  # 3:4
    (1280, 768),  # 16:9
    (768, 1280),  # 9:16
]


def map_to_standard_shapes(target_width, target_height):
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height


def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg



class HunyuanDiTPipeline_DoE_combined(DiffusionPipeline, SD3LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = [
        "feature_extractor",
        "text_encoder_2",
        "tokenizer_2",
        "text_encoder",
        "tokenizer",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "prompt_embeds_2",
        "negative_prompt_embeds_2",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: BertModel,
        tokenizer: BertTokenizer,
        transformer: HunyuanDiT2DModel,
        scheduler: DDPMScheduler,
        feature_extractor: CLIPImageProcessor,
        controlnet: HunyuanDiT2DControlNetModel,
        text_encoder_2=T5EncoderModel,
        tokenizer_2=MT5Tokenizer,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            text_encoder_2=text_encoder_2,
            controlnet=controlnet,
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor  = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=self.vae.config.latent_channels, 
                                                 do_resize=False, do_normalize=False, do_binarize=False, do_convert_grayscale=True)
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        prompt_embeds_2=None,
        negative_prompt_embeds_2=None,
        prompt_attention_mask_2=None,
        negative_prompt_attention_mask_2=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        #   currently always use text encoder 1, but _2 (T5) is optional, hence not checked those here
        if prompt_embeds == None or negative_prompt_embeds == None or prompt_attention_mask == None or negative_prompt_attention_mask == None:
            raise ValueError(f"All prompt embeds must be provided.")

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        strength: float = 0.6,
        mask_image: PipelineImageInput = None,
        mask_cutoff: float = 1.0,

        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 5.0,
        guidance_rescale: float = 0.0,
        guidance_cutoff: float = 1.0,

        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        original_size: Optional[Tuple[int, int]] = (1024, 1024),
        target_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        use_resolution_binning: bool = True,
        
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,


        centre_latents: Optional[bool] = False,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,

    ):
        doDiffDiff = True if (image and mask_image) else False
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        height = int((height // 16) * 16)
        width = int((width // 16) * 16)

        if use_resolution_binning and (height, width) not in SUPPORTED_SHAPE:
            width, height = map_to_standard_shapes(width, height)
            height = int(height)
            width = int(width)
            logger.warning(f"Reshaped to (height, width)=({height}, {width}), Supported shapes are {SUPPORTED_SHAPE}")
            
            
        # 0.01 repeat prompt embeds to match num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        prompt_embeds_2 = prompt_embeds_2.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
        negative_prompt_embeds_2 = negative_prompt_embeds_2.repeat(num_images_per_prompt, 1, 1)

        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask_2 = prompt_attention_mask_2.repeat(num_images_per_prompt, 1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        negative_prompt_attention_mask_2 = negative_prompt_attention_mask_2.repeat(num_images_per_prompt, 1)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters

        device = self._execution_device

        # prepare controlnet image
        if self.controlnet != None and control_image != None:
            if isinstance(self.controlnet, HunyuanDiT2DControlNetModel):
                control_image = self.image_processor.preprocess(control_image, height=height, width=width).to('cuda').to(torch.float16)

                control_image = self.vae.encode(control_image).latent_dist.sample()
                control_image = control_image * self.vae.config.scaling_factor

                control_image = control_image.repeat_interleave(num_images_per_prompt, dim=0)   #repeat interleave handles multiple control images

                if self.do_classifier_free_guidance:
                    control_image = torch.cat([control_image] * 2)


        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 5. Prepare latent variables
        # input noise is already mixed with colour, sized correctly (num_images_per_prompt), moved to device

        if image is not None:
            noise = latents

            # 4. Prepare timesteps
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

            # 3. Preprocess image
            image = self.image_processor.preprocess(image, height=height, width=width).to('cuda').to(torch.float16)
            image_latents = self.vae.encode(image).latent_dist.sample(generator)
            image_latents *= self.vae.config.scaling_factor * self.scheduler.init_noise_sigma
            image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)

            # add noise to image latents
            if strength < 1.0:
                ts = torch.tensor([timesteps[0]], device='cuda')
                ts = ts[:1].repeat(num_images_per_prompt)
                latents = self.scheduler.add_noise(image_latents, noise, ts)

            if mask_image is not None:
                # 5.1. Prepare masked latent variables
                mask = self.mask_processor.preprocess(mask_image.resize((width//8, height//8))).to(device='cuda', dtype=torch.float16)
               
        else:
            timesteps = self.scheduler.timesteps

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7 create image_rotary_emb, style embedding & time ids
        grid_height = height // 8 // self.transformer.config.patch_size
        grid_width = width // 8 // self.transformer.config.patch_size
        base_size = 512 // 8 // self.transformer.config.patch_size
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size)
        image_rotary_emb = get_2d_rotary_pos_embed(
            self.transformer.inner_dim // self.transformer.num_heads, grid_crops_coords, (grid_height, grid_width)
        )

        style = torch.tensor([0], device='cuda')

        target_size = target_size or (height, width)
        add_time_ids = list(original_size + target_size + crops_coords_top_left)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device='cuda')

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])
            add_time_ids = torch.cat([add_time_ids] * 2, dim=0)
            style = torch.cat([style] * 2, dim=0)

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device)
        prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
        add_time_ids = add_time_ids.to(dtype=prompt_embeds.dtype, device=device).repeat(num_images_per_prompt, 1)
        style = style.to(device=device).repeat(num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if doDiffDiff and float((i+1) / self._num_timesteps) <= mask_cutoff:
                    tmask = (mask >= float((i+1) / self._num_timesteps))
                    ts = torch.tensor([t], device='cuda')
                    ts = ts[:1].repeat(num_images_per_prompt)
                    init_latents_proper = self.scheduler.add_noise(image_latents, noise, ts)
                    latents = (init_latents_proper * ~tmask) + (latents * tmask)

                if float((i+1) / len(timesteps)) > guidance_cutoff and self._guidance_scale != 1.0:
                    self._guidance_scale = 1.0
                    prompt_embeds = prompt_embeds[num_images_per_prompt:]#.unsqueeze(0)
                    prompt_attention_mask = prompt_attention_mask[num_images_per_prompt:]
                    prompt_embeds_2 = prompt_embeds_2[num_images_per_prompt:]
                    prompt_attention_mask_2 = prompt_attention_mask_2[num_images_per_prompt:]
                    add_time_ids = add_time_ids[num_images_per_prompt:]
                    style = style[num_images_per_prompt:]

                    if self.controlnet != None:
                        control_image = control_image[num_images_per_prompt:]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
                t_expand = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )

                if self.controlnet != None and float(i / len(timesteps)) >= control_guidance_start and float(i / len(timesteps)) <= control_guidance_end:
                    # controlnet(s) inference
                    control_block_samples = self.controlnet(
                        latent_model_input,
                        t_expand,
                        encoder_hidden_states=prompt_embeds,
                        text_embedding_mask=prompt_attention_mask,
                        encoder_hidden_states_t5=prompt_embeds_2,
                        text_embedding_mask_t5=prompt_attention_mask_2,
                        image_meta_size=add_time_ids,
                        style=style,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                        controlnet_cond=control_image,
                        conditioning_scale=controlnet_conditioning_scale,
                    )[0]
                else:
                    control_block_samples = None

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,
                    t_expand,
                    encoder_hidden_states=prompt_embeds,
                    text_embedding_mask=prompt_attention_mask,
                    encoder_hidden_states_t5=prompt_embeds_2,
                    text_embedding_mask_t5=prompt_attention_mask_2,
                    image_meta_size=add_time_ids,
                    style=style,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
#                    cross_attention_kwargs=self.cross_attention_kwargs,
                    controlnet_block_samples=control_block_samples,
                )[0]


                noise_pred, _ = noise_pred.chunk(2, dim=1)
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

#   might have value as a option
                if centre_latents == True:
                    for b in range(len(latents)):
                        for c in range(4):
                            latents[b][c] -= latents[b][c].mean()


                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                    negative_prompt_embeds_2 = callback_outputs.pop(
                        "negative_prompt_embeds_2", negative_prompt_embeds_2
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if doDiffDiff and 1.0 <= mask_cutoff:
            tmask = (mask >= 1.0)
            latents = (image_latents * ~tmask) + (latents * tmask)

        # Offload all models
        self.maybe_free_model_hooks()

        return latents
