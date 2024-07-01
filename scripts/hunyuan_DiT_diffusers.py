import math
import torch
import gc
import json
import numpy as np
import os

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

from PIL import Image
#workaround for unnecessary flash_attn requirement for Florence-2
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)

import customStylesListHY as styles


class HunyuanStorage:
    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    positive_embeds_1 = None
    negative_embeds_1 = None
    prompt_attention_1 = None
    negative_attention_1 = None
    prompt_embeds_2 = None
    negative_embeds_2 = None
    prompt_attention_2 = None
    negative_attention_2 = None
    karras = False
    useT5 = True
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    i2iAllSteps = False
    lora = None
    lora_scale = 1.0
    last_batch = 0

## from huggingace.co/Tencent-Hunyuan/HYDiT-LoRA
def load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale):
    num_layers = 40
    for i in range(num_layers):
        Wqkv = torch.matmul(lora_state_dict[f"base_model.model.blocks.{i}.attn1.Wqkv.lora_B.weight"], 
                            lora_state_dict[f"base_model.model.blocks.{i}.attn1.Wqkv.lora_A.weight"]) 
        q, k, v = torch.chunk(Wqkv, 3, dim=0)
        q *= lora_scale
        k *= lora_scale
        v *= lora_scale

        transformer_state_dict[f"blocks.{i}.attn1.to_q.weight"] += q.to('cpu')
        transformer_state_dict[f"blocks.{i}.attn1.to_k.weight"] += k.to('cpu')
        transformer_state_dict[f"blocks.{i}.attn1.to_v.weight"] += v.to('cpu')

        out_proj = torch.matmul(lora_state_dict[f"base_model.model.blocks.{i}.attn1.out_proj.lora_B.weight"], 
                                lora_state_dict[f"base_model.model.blocks.{i}.attn1.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.to_out.0.weight"] += lora_scale * out_proj.to('cpu')

        q_proj = torch.matmul(lora_state_dict[f"base_model.model.blocks.{i}.attn2.q_proj.lora_B.weight"], 
                              lora_state_dict[f"base_model.model.blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_q.weight"] += lora_scale * q_proj.to('cpu')

        kv_proj = torch.matmul(lora_state_dict[f"base_model.model.blocks.{i}.attn2.kv_proj.lora_B.weight"], 
                               lora_state_dict[f"base_model.model.blocks.{i}.attn2.kv_proj.lora_A.weight"])
        k, v = torch.chunk(kv_proj, 2, dim=0)
        transformer_state_dict[f"blocks.{i}.attn2.to_k.weight"] += k.to('cpu')
        transformer_state_dict[f"blocks.{i}.attn2.to_v.weight"] += v.to('cpu')

        out_proj = torch.matmul(lora_state_dict[f"base_model.model.blocks.{i}.attn2.out_proj.lora_B.weight"], 
                                lora_state_dict[f"base_model.model.blocks.{i}.attn2.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn2.to_out.0.weight"] += lora_scale * out_proj.to('cpu')
    
    q_proj = torch.matmul(lora_state_dict["base_model.model.pooler.q_proj.lora_B.weight"], lora_state_dict["base_model.model.pooler.q_proj.lora_A.weight"])
    transformer_state_dict["time_extra_emb.pooler.q_proj.weight"] += lora_scale * q_proj.to('cpu')
    
    return transformer_state_dict



from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel
#from diffusers.models.controlnet_hunyuan import HunyuanDiT2DControlNetModel#, HunyuanDiT2DMultiControlNetModel

from scripts.HY_pipeline import HunyuanDiTPipeline_DoE_combined

#from diffusers import HunyuanDiTPipeline
from diffusers import AutoencoderKL, HunyuanDiT2DModel

from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging

import argparse
import pathlib
from pathlib import Path
import sys

current_file_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_file_path))


# modules/infotext_utils.py
def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text

    return json.dumps(text, ensure_ascii=False)

# modules/processing.py
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, steps, seed, scheduler, width, height, controlNetSettings, state):
    karras = " : Karras" if state[1] == True else ""
    generation_params = {
        "T5":           '✓' if state[0] else '✗', #2713, 2717
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps": steps,
        "CFG": f"{guidance_scale} ({guidance_rescale})",
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "controlNet": controlNetSettings,
    }

    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
    
    noise_text = f"\nInitial noise: {HunyuanStorage.noiseRGBA}" if HunyuanStorage.noiseRGBA[3] != 0.0 else ""

    return f"Model: {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(model, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale,
            num_steps, sampling_seed, num_images, scheduler, style, i2iSource, i2iDenoise, maskSource, 
            controlNet, controlNetImage, controlNetStrength, controlNetStart, controlNetEnd, 
            *args):

    torch.set_grad_enabled(False)

    volatileState = [HunyuanStorage.useT5, HunyuanStorage.karras]

    if controlNet != 0 and controlNetImage != None and controlNetStrength > 0.0:
        controlNetImage = controlNetImage.resize((width, height))
        useControlNet = ['Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny', 
                         'Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Depth', 
                         'Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Pose'][controlNet-1]
    else:
        controlNetStrength = 0.0
        useControlNet = None
        
    useControlNet = None

    if i2iSource != None:
        if HunyuanStorage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)
        i2iSource = i2iSource.resize((width, height))
    else:
        i2iDenoise = 1.0
        maskSource = None

    if maskSource != None:
        maskSource = maskSource.resize((int(width/8), int(height/8)))
    else:
        maskCutOff = 1.0


    #   double prompt, automatic support, no longer needs button to enable
    split_positive = positive_prompt.split('|')
    pc = len(split_positive)
    if pc == 1:
        positive_prompt_1 = split_positive[0].strip()
        positive_prompt_2 = positive_prompt_1
    elif pc >= 2:
        positive_prompt_1 = split_positive[0].strip()
        positive_prompt_2 = split_positive[1].strip()
        
    split_negative = negative_prompt.split('|')
    nc = len(split_negative)
    if nc == 1:
        negative_prompt_1 = split_negative[0].strip()
        negative_prompt_2 = negative_prompt_1
    elif nc >= 2:
        negative_prompt_1 = split_negative[0].strip()
        negative_prompt_2 = split_negative[1].strip()

    if style != 0:  #better to rebuild stored prompt from _1,_2,_3 so random changes at end /whitespace effect nothing
        positive_prompt_1 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_1)
        positive_prompt_2 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_2)
        negative_prompt_1 = styles.styles_list[style][2] + negative_prompt_1
        negative_prompt_2 = styles.styles_list[style][2] + negative_prompt_2

    combined_positive = positive_prompt_1 + " |\n" + positive_prompt_2
    combined_negative = negative_prompt_1 + " |\n" + negative_prompt_2

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    HunyuanStorage.lastSeed = fixed_seed

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    #first: tokenize and text_encode
    useCachedEmbeds = (HunyuanStorage.lastPrompt == combined_positive and 
                       HunyuanStorage.lastNegative == combined_negative and
                       HunyuanStorage.last_batch == num_images)
    if useCachedEmbeds:
        print ("Skipping text encoders and tokenizers.")
        #   nothing to do
    else:
        #   tokenize 1
        tokenizer = BertTokenizer.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='tokenizer',
            torch_dtype=torch.float16,
            )
        #   positive
        text_inputs = tokenizer(
            positive_prompt_1,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        positive_text_input_ids = text_inputs.input_ids.to('cuda')
        positive_attention_1 = text_inputs.attention_mask.to('cuda')
        #   negative
        text_inputs = tokenizer(
            negative_prompt_1,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        negative_text_input_ids = text_inputs.input_ids.to('cuda')
        negative_attention_1 = text_inputs.attention_mask.to('cuda')

        del tokenizer, text_inputs
        #end tokenize 1

        #   text encode 1
        text_encoder = BertModel.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='text_encoder',
            torch_dtype=torch.float16,
            ).to('cuda')

        prompt_embeds = text_encoder(
            positive_text_input_ids,
            attention_mask=positive_attention_1,
        )
        bs_embed, seq_len, _ = prompt_embeds[0].shape
        positive_embeds_1 = prompt_embeds[0].repeat(1, num_images, 1)
        positive_embeds_1 = positive_embeds_1.view(bs_embed * num_images, seq_len, -1)
        positive_attention_1 = positive_attention_1.repeat(num_images, 1)

        prompt_embeds = text_encoder(
            negative_text_input_ids,
            attention_mask=negative_attention_1,
        )
        seq_len = prompt_embeds[0].shape[1]
        negative_embeds_1 = prompt_embeds[0].repeat(1, num_images, 1)
        negative_embeds_1 = negative_embeds_1.view(num_images, seq_len, -1)
        negative_attention_1 = negative_attention_1.repeat(num_images, 1)

        del text_encoder, prompt_embeds
        #end text_encode 1

        gc.collect()
        torch.cuda.empty_cache()

        if HunyuanStorage.useT5 == True:
            #   tokenize 2
            tokenizer = MT5Tokenizer.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_2',
                torch_dtype=torch.float16,
                )
            #   positive
            text_inputs = tokenizer(
                positive_prompt_2,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            positive_text_input_ids = text_inputs.input_ids.to('cuda')
            positive_attention_2 = text_inputs.attention_mask.to('cuda')
            #   negative
            text_inputs = tokenizer(
                negative_prompt_2,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            negative_text_input_ids = text_inputs.input_ids.to('cuda')
            negative_attention_2 = text_inputs.attention_mask.to('cuda')

            del tokenizer, text_inputs
            #end tokenize 2

            #   text encode 2
            text_encoder = T5EncoderModel.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder_2',
                torch_dtype=torch.float16,
                device_map='auto'
                )

            prompt_embeds = text_encoder(
                positive_text_input_ids,
                attention_mask=positive_attention_2,
            )
            bs_embed, seq_len, _ = prompt_embeds[0].shape
            positive_embeds_2 = prompt_embeds[0].repeat(1, num_images, 1)
            positive_embeds_2 = positive_embeds_2.view(bs_embed * num_images, seq_len, -1)
            positive_attention_2 = positive_attention_2.repeat(num_images, 1)

            prompt_embeds = text_encoder(
                negative_text_input_ids,
                attention_mask=negative_attention_2,
            )
            seq_len = prompt_embeds[0].shape[1]
            negative_embeds_2 = prompt_embeds[0].repeat(1, num_images, 1)
            negative_embeds_2 = negative_embeds_2.view(num_images, seq_len, -1)
            negative_attention_2 = negative_attention_2.repeat(num_images, 1)

            del text_encoder, prompt_embeds
            #end text_encode 2
        else:
            #256 is tokenizer max length from config; 2048 is transformer joint_attention_dim from its config
            positive_embeds_2    = torch.zeros((num_images, 256, 2048))
            positive_attention_2 = torch.zeros((num_images, 256))
            negative_embeds_2    = torch.zeros((num_images, 256, 2048))
            negative_attention_2 = torch.zeros((num_images, 256))

        HunyuanStorage.positive_embeds_1    = positive_embeds_1.to('cpu')
        HunyuanStorage.positive_attention_1 = positive_attention_1.to('cpu')
        HunyuanStorage.negative_embeds_1    = negative_embeds_1.to('cpu')
        HunyuanStorage.negative_attention_1 = negative_attention_1.to('cpu')
        HunyuanStorage.positive_embeds_2    = positive_embeds_2.to('cpu')
        HunyuanStorage.positive_attention_2 = positive_attention_2.to('cpu')
        HunyuanStorage.negative_embeds_2    = negative_embeds_2.to('cpu')
        HunyuanStorage.negative_attention_2 = negative_attention_2.to('cpu')

        del positive_embeds_1, negative_embeds_1, positive_attention_1, negative_attention_1
        del positive_embeds_2, negative_embeds_2, positive_attention_2, negative_attention_2

        HunyuanStorage.lastPrompt = combined_positive
        HunyuanStorage.lastNegative = combined_negative
        HunyuanStorage.last_batch = num_images

    gc.collect()
    torch.cuda.empty_cache()

    #second: transformer/VAE
    source = "Tencent-Hunyuan/" + model
    transformer = HunyuanDiT2DModel.from_pretrained(
        source,
        local_files_only=False, cache_dir=".//models//diffusers//",
        subfolder='transformer',
        torch_dtype=torch.float16,
        )

#    controlnet=HunyuanDiT2DControlNetModel.from_pretrained(useControlNet, cache_dir=".//models//diffusers//", torch_dtype=torch.float16) if useControlNet else None
    controlnet = None

    pipe = HunyuanDiTPipeline_DoE_combined.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        local_files_only=False, cache_dir=".//models//diffusers//",
        transformer=transformer,
        feature_extractor=None,
        torch_dtype=torch.float16,
        tokenizer=None,
        text_encoder=None,
        tokenizer_2=None,
        text_encoder_2=None,
        use_safetensors=True,
#        controlnet=controlnet,
        )
    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
       
#    pipe.transformer.enable_forward_chunking(chunk_size=1, dim=1)      #>= 0.28.2 ?
    pipe.vae.enable_tiling(True)
#    pipe.enable_attention_slicing()

    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    shape = (
        num_images,
        pipe.transformer.config.in_channels,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )

    latents = randn_tensor(shape, generator=generator, dtype=torch.float16).to('cuda').to(torch.float16)
    #   colour the initial noise
    if HunyuanStorage.noiseRGBA[3] != 0.0:
        nr = HunyuanStorage.noiseRGBA[0] ** 0.5
        ng = HunyuanStorage.noiseRGBA[1] ** 0.5
        nb = HunyuanStorage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(np.full((8,8), (nr), dtype=np.float32))
        imageG = torch.tensor(np.full((8,8), (ng), dtype=np.float32))
        imageB = torch.tensor(np.full((8,8), (nb), dtype=np.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = pipe.vae.encode(image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor * pipe.scheduler.init_noise_sigma
        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

        for b in range(len(latents)):
            for c in range(4):
                latents[b][c] -= latents[b][c].mean()

#        latents += image_latents * HunyuanStorage.noiseRGBA[3]
        torch.lerp (latents, image_latents, HunyuanStorage.noiseRGBA[3], out=latents)

        del imageR, imageG, imageB, image, image_latents
    #   end: colour the initial noise

#   load in LoRA

    if HunyuanStorage.lora and HunyuanStorage.lora != "(None)" and HunyuanStorage.lora_scale != 0.0:
        lorafile = ".//models/diffusers//HunyuanLora//" + HunyuanStorage.lora + ".safetensors"

        from safetensors import safe_open
        lora_state_dict = {}
        with safe_open(lorafile, framework="pt", device=0) as f:
            for k in f.keys():
                lora_state_dict[k] = f.get_tensor(k)
                
        if "base_model.model.blocks.0.attn1.Wqkv.lora_A.weight" in lora_state_dict:  #   needs converting from Tencent HY Lora
            transformer_state_dict = pipe.transformer.state_dict()
            transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=HunyuanStorage.lora_scale)
            pipe.transformer.load_state_dict(transformer_state_dict)
        elif "lora_unet_blocks_0_attn1_Wqkv.alpha" in lora_state_dict:                       #   needs converting from safetensors lora
            print ("Unsupported LoRA type.")
            return gr.Button.update(value='Generate', variant='primary', interactive=True), None
        else:                                                       #   already converted, or incompatible
            try:
                logging.set_verbosity(logging.ERROR)
                pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=HunyuanStorage.lora)
                logging.set_verbosity(logging.WARN)
            except:
                print ("Failed: LoRA: " + lorafile)
                return gr.Button.update(value='Generate', variant='primary', interactive=True), None
        del lora_state_dict
   
    if scheduler == 'DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DEIS':
        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM++ 2M':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "DPM++ 2M SDE":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='sde-dpmsolver++')
    elif scheduler == 'DPM':
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'DPM SDE':
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == 'Euler A':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "SA-solver":
        pipe.scheduler = SASolverScheduler.from_config(pipe.scheduler.config, algorithm_type='data_prediction')
    elif scheduler == 'UniPC':
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#   else uses default set by model (DDPM)

    if hasattr(pipe.scheduler.config, 'use_karras_sigmas'):
        pipe.scheduler.config.use_karras_sigmas = HunyuanStorage.karras

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = pipe(
            image=i2iSource,
            strength=i2iDenoise,
            mask_image=maskSource,

            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            prompt_embeds=HunyuanStorage.positive_embeds_1.to('cuda').to(torch.float16),
            negative_prompt_embeds=HunyuanStorage.negative_embeds_1.to('cuda').to(torch.float16),
            prompt_attention_mask=HunyuanStorage.positive_attention_1.to('cuda').to(torch.float16),
            negative_prompt_attention_mask=HunyuanStorage.negative_attention_1.to('cuda').to(torch.float16),
            prompt_embeds_2=HunyuanStorage.positive_embeds_2.to('cuda').to(torch.float16),
            negative_prompt_embeds_2=HunyuanStorage.negative_embeds_2.to('cuda').to(torch.float16),
            prompt_attention_mask_2=HunyuanStorage.positive_attention_2.to('cuda').to(torch.float16),
            negative_prompt_attention_mask_2=HunyuanStorage.negative_attention_2.to('cuda').to(torch.float16),
            num_images_per_prompt=num_images,
            output_type="pil",
            generator=generator,
            use_resolution_binning=False,
            latents=latents,

            control_image=controlNetImage, 
            controlnet_conditioning_scale=controlNetStrength,  
            control_guidance_start=controlNetStart,
            control_guidance_end=controlNetEnd,
            
#            cross_attention_kwargs={"scale": HunyuanStorage.lora_scale }    #   currently does nothing - HYDiT forward pass doesn't take this input
        )

    del pipe, generator, latents, controlNetImage

    gc.collect()
    torch.cuda.empty_cache()

    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}, step range: {controlNetStart}-{controlNetEnd}"

    result = []
    for image in output:
        info=create_infotext(
            model, 
            combined_positive, combined_negative,
            guidance_scale, guidance_rescale, num_steps, 
            fixed_seed, scheduler,
            width, height, 
            useControlNet, volatileState)

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            combined_positive,
            opts.samples_format,
            info
        )
        fixed_seed += 1

    del output
    gc.collect()
    torch.cuda.empty_cache()

    return gr.Button.update(value='Generate', variant='primary', interactive=True), result


def on_ui_tabs():
    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\HunyuanLora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras

    loras = buildLoRAList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gr.Dropdown.update(choices=loras)
   
    def getGalleryIndex (evt: gr.SelectData):
        HunyuanStorage.galleryIndex = evt.index

    def reuseLastSeed ():
        return HunyuanStorage.lastSeed + HunyuanStorage.galleryIndex
        
    def randomSeed ():
        return -1

    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = 32 * (image.size[0] // 32)
            h = 32 * (image.size[1] // 32)
        return [w, h]

    def i2iImageFromGallery (gallery):
        try:
            newImage = gallery[HunyuanStorage.galleryIndex][0]['name'].split('?')
            return newImage[0]
        except:
            return None

    def toggleC2P ():
        HunyuanStorage.captionToPrompt ^= True
        return gr.Button.update(variant=['secondary', 'primary'][HunyuanStorage.captionToPrompt])
    def toggleKarras ():
        HunyuanStorage.karras ^= True
        return gr.Button.update(variant=['secondary', 'primary'][HunyuanStorage.karras],
                                value=['\U0001D542', '\U0001D40A'][HunyuanStorage.karras])

    def toggleT5 ():
        HunyuanStorage.lastPrompt = None
        HunyuanStorage.lastNegative = None
        HunyuanStorage.useT5 ^= True
        return gr.Button.update(variant=['secondary', 'primary'][HunyuanStorage.useT5])
    def toggleAS ():
        HunyuanStorage.i2iAllSteps ^= True
        return gr.Button.update(variant=['secondary', 'primary'][HunyuanStorage.i2iAllSteps])

    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.float32, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True)
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            if p != prompts[-1]:
                result += ' | \n'
            del parsed_answer

        del model, processor

        if HunyuanStorage.captionToPrompt:
            return result
        else:
            return originalPrompt

    def toggleGenerate (R, G, B, A, lora, scale):
        HunyuanStorage.noiseRGBA = [R, G, B, A]
        HunyuanStorage.lora = lora
        HunyuanStorage.lora_scale = scale# if lora != "(None)" else 1.0
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as hunyuandit_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    model = gr.Dropdown([#'HunyuanDiT-v1.2-Diffusers-Distilled',
                                         #'HunyuanDiT-v1.2-Diffusers',
                                         'HunyuanDiT-v1.1-Diffusers-Distilled',
                                         'HunyuanDiT-v1.1-Diffusers',
                                         'HunyuanDiT-Diffusers-Distilled',
                                         'HunyuanDiT-Diffusers',
                                         ], label='Model', value='HunyuanDiT-v1.1-Diffusers', type='value')

                    T5 = ToolButton(value="T5", variant='primary', tooltip="use T5 text encoder")
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
                with gr.Row():
                    positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=1.1)
                    scheduler = gr.Dropdown(["DDPM",
                                             "DEIS",
                                             "DPM++ 2M",
                                             "DPM++ 2M SDE",
                                             "DPM",
                                             "DPM SDE",
                                             "Euler",
                                             "Euler A",
                                             "SA-solver",
                                             "UniPC",
                                             ],
                        label='Sampler', value="SA-solver", type='value', scale=0)

                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=1.1)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gr.Slider(label='Height', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_height")

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.5, value=4, scale=2)
                    guidance_rescale = gr.Slider(label='rescale CFG', minimum=0, maximum=1, step=0.01, value=0, scale=2)
                with gr.Row():
                    steps = gr.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gr.Row(equal_height=True):
                    lora = gr.Dropdown([x for x in loras], label='LoRA', value="(None)", type='value', multiselect=False, scale=1)
                    refresh = ToolButton(value='\U0001f504')
                    scale = gr.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01)

                with gr.Accordion(label='the colour of noise', open=False):
                    with gr.Row():
                        initialNoiseR = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gr.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gr.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')

                with gr.Accordion(label='ControlNet', open=False, visible=False):
                    with gr.Row():
                        CNSource = gr.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            CNMethod = gr.Dropdown(['(None)', 'canny', 'depth', 'pose'], label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gr.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gr.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gr.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        maskSource = gr.Image(label='source mask', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            with gr.Row():
                                i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value='AS')
                            i2iSetWH = gr.Button(value='Set safe Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')
                            with gr.Row():
                                i2iCaption = gr.Button(value='Caption this image (Florence-2)', scale=9)
                                toPrompt = ToolButton(value='P', variant='secondary')
                            maskCut = gr.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)

                ctrls = [model, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, steps, sampling_seed,
                         batch_size, scheduler, style, i2iSource, i2iDenoise, maskSource, CNMethod, CNSource, CNStrength, CNStart, CNEnd]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height="80vh",
                                            show_label=False, object_fit='contain', visible=True, columns=3, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=positive_prompt,
                        source_image_component=output_gallery,
                    ))


        refresh.click(refreshLoRAs, inputs=[], outputs=[lora])
        karras.click(toggleKarras, inputs=[], outputs=karras)
        T5.click(toggleT5, inputs=[], outputs=T5)
        AS.click(toggleAS, inputs=[], outputs=AS)
        swapper.click(fn=None, _js="function(){switchWidthHeight('Hunyuan-DiT')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])#outputs=[positive_prompt]
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, output_gallery])
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button])

    return [(hunyuandit_block, "Hunyuan-DiT", "hunyuan")]

script_callbacks.on_ui_tabs(on_ui_tabs)

