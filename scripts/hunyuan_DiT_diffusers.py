####    TODO:
##      PAG ?


class HunyuanStorage:
    ModuleReload = False
    lastSeed = -1
    galleryIndex = 0
    lastPrompt = None
    lastNegative = None
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    lora = None
    lora_scale = 1.0
    
    teCLIP = None
    teT5 = None
    pipe = None
    lastTR = None
    lastControlNet = None
    loadedLora = False
    transformerStateDict = None

    locked = False      #   for preventing changes to the following volatile state while generating
    noUnload = False
    karras = False
    useT5 = True
    centreLatents = False
    zeroSNR = False
    i2iAllSteps = False
    sharpNoise = False

import gc
import gradio
import math
import numpy
import os
import torch
import torchvision.transforms.functional as TF
try:
    from importlib import reload
    HunyuanStorage.ModuleReload = True
except:
    HunyuanStorage.ModuleReload = False

##  from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from diffusers import AutoencoderKL, HunyuanDiT2DModel
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging
from diffusers.models import HunyuanDiT2DControlNetModel#, HunyuanDiT2DMultiControlNetModel

##  for Florence-2, including workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from transformers import AutoProcessor, AutoModelForCausalLM 

##  my stuff
import customStylesListHY as styles
import scripts.HY_pipeline as pipeline


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




# modules/processing.py
def create_infotext(model, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, guidance_cutoff, steps, seed, scheduler, width, height, loraSettings, controlNetSettings):
    karras = " : Karras" if HunyuanStorage.karras == True else ""
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps": steps,
        "CFG": f"{guidance_scale} ({guidance_rescale}) [{guidance_cutoff}]",
        "LoRA"          :   loraSettings,
        "controlNet": controlNetSettings,
        "T5":           '✓' if HunyuanStorage.useT5 else '✗', #2713, 2717
        "center latents":   '✓' if HunyuanStorage.centreLatents else '✗',
        "zSNR":             '✓' if HunyuanStorage.zeroSNR else '✗',
    }

    prompt_text = f"Prompt: {positive_prompt}\n"
    prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    
    noise_text = f"\nInitial noise: {HunyuanStorage.noiseRGBA}" if HunyuanStorage.noiseRGBA[3] != 0.0 else ""

    return f"Model: {model}\n{prompt_text}{generation_params_text}{noise_text}"

def predict(model, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, guidance_cutoff,
            num_steps, sampling_seed, num_images, scheduler, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCutOff, 
            controlNet, controlNetImage, controlNetStrength, controlNetStart, controlNetEnd, 
            *args):

    logging.set_verbosity(logging.ERROR)    #   minor issue in config.json causes console spam, and LoRAs

    torch.set_grad_enabled(False)

    if controlNet != 0 and controlNetImage != None and controlNetStrength > 0.0:
        useControlNet = ['Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny', 
                         'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Depth', 
                         'Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Pose'][controlNet-1]

    else:
        useControlNet = None
        controlNetImage = None
        controlNetStrength = 0.0

    ####    check img2img
    if i2iSource == None:
        maskType = 0
        i2iDenoise = 1
    if maskSource == None:
        maskType = 0
    if HunyuanStorage.i2iAllSteps == True:
        num_steps = int(num_steps / i2iDenoise)
        
    match maskType:
        case 0:     #   'none'
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0
        case 1:     #   'image'
            maskSource = maskSource['image']
        case 2:     #   'drawn'
            maskSource = maskSource['mask']
        case _:
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0

    if maskBlur > 0:
        maskSource = TF.gaussian_blur(maskSource, 1+2*maskBlur*8)
    ####    end check img2img


    #   double prompt, automatic support, no longer needs button to enable
    def promptSplit (prompt):
        split_prompt = prompt.split('|')
        prompt_1 = split_prompt[0].strip()
        prompt_2 = split_prompt[1].strip() if len(split_prompt) > 1 else prompt_1
        return prompt_1, prompt_2

    positive_prompt_1, positive_prompt_2 = promptSplit (positive_prompt)
    negative_prompt_1, negative_prompt_2 = promptSplit (negative_prompt)

    if style != 0:  #better to rebuild stored prompt from _1,_2,_3 so random changes at end /whitespace effect nothing
        positive_prompt_1 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_1)
        positive_prompt_2 = styles.styles_list[style][1].replace("{prompt}", positive_prompt_2)
        negative_prompt_1 = negative_prompt_1 + styles.styles_list[style][2]
        negative_prompt_2 = negative_prompt_2 + styles.styles_list[style][2]

    combined_positive = positive_prompt_1 + " | \n" + positive_prompt_2
    combined_negative = negative_prompt_1 + " | \n" + negative_prompt_2

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    HunyuanStorage.lastSeed = fixed_seed

    #first: tokenize and text_encode
    useCachedEmbeds = (HunyuanStorage.lastPrompt   == combined_positive and 
                       HunyuanStorage.lastNegative == combined_negative)
    if useCachedEmbeds:
        print ("Hunyuan: Skipping text encoders and tokenizers.")
        #   nothing to do
    else:
        #   tokenize 1
        tokenizer = BertTokenizer.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='tokenizer',
            torch_dtype=torch.float16,
            )
        #   positive and negative in same run
        text_inputs = tokenizer(
            [positive_prompt_1, negative_prompt_1],
            padding='max_length',       max_length=77,      truncation=True,
            return_attention_mask=True, return_tensors="pt",
        )

        positive_attention_1 = text_inputs.attention_mask[0:1]
        negative_attention_1 = text_inputs.attention_mask[1:]

        text_input_ids = text_inputs.input_ids
        attention = text_inputs.attention_mask

        del tokenizer, text_inputs
        #end tokenize 1

        #   text encode 1
        if HunyuanStorage.teCLIP == None:
            HunyuanStorage.teCLIP = BertModel.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='text_encoder',
                torch_dtype=torch.float16,
                ).to('cuda')

        prompt_embeds = HunyuanStorage.teCLIP(
            text_input_ids.to('cuda'),
            attention_mask=attention.to('cuda'),
        )[0]
        positive_embeds_1 = prompt_embeds[0].unsqueeze(0)
        negative_embeds_1 = prompt_embeds[1].unsqueeze(0)

        del prompt_embeds
        if HunyuanStorage.noUnload == False:
            HunyuanStorage.teCLIP = None
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
            tokens = tokenizer(
                [positive_prompt_2, negative_prompt_2],
                padding=True, max_length=256, truncation=True,
                return_attention_mask=True, return_tensors="pt",
            )

            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask

            positive_attention_2 = attention_mask[0:1]
            negative_attention_2 = attention_mask[1:]

            del tokenizer#, text_inputs
            #end tokenize 2

            #   text encode 2
            if HunyuanStorage.teT5 == None:
                if HunyuanStorage.noUnload == True:     #   will keep model loaded
                    device_map = {  #   how to find which blocks are most important? if any?
                        'shared': 0,
                        'encoder.embed_tokens': 0,
                        'encoder.block.0': 'cpu',   'encoder.block.1': 'cpu',   'encoder.block.2': 'cpu',   'encoder.block.3': 'cpu', 
                        'encoder.block.4': 'cpu',   'encoder.block.5': 'cpu',   'encoder.block.6': 'cpu',   'encoder.block.7': 'cpu', 
                        'encoder.block.8': 'cpu',   'encoder.block.9': 'cpu',   'encoder.block.10': 'cpu',  'encoder.block.11': 'cpu', 
                        'encoder.block.12': 'cpu',  'encoder.block.13': 'cpu',  'encoder.block.14': 'cpu',  'encoder.block.15': 'cpu', 
                        'encoder.block.16': 'cpu',  'encoder.block.17': 'cpu',  'encoder.block.18': 'cpu',  'encoder.block.19': 'cpu', 
                        'encoder.block.20': 0,      'encoder.block.21': 0,      'encoder.block.22': 0,      'encoder.block.23': 0, 
                        'encoder.final_layer_norm': 0, 
                        'encoder.dropout': 0
                    }
                else:                               #   will delete model after use
                    device_map = 'auto'

                print ("Hunyuan: loading T5 ...", end="\r", flush=True)
                HunyuanStorage.teT5 = T5EncoderModel.from_pretrained(
                    "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                    local_files_only=False, cache_dir=".//models//diffusers//",
                    subfolder='text_encoder_2',
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )

            print ("Hunyuan: encoding prompt (T5) ...", end="\r", flush=True)

            prompt_embeds = HunyuanStorage.teT5(
                input_ids.to('cuda'),
                attention_mask=attention_mask.to('cuda'),
            )[0]
            positive_embeds_2 = prompt_embeds[0].unsqueeze(0)
            negative_embeds_2 = prompt_embeds[1].unsqueeze(0)

            print ("Hunyuan: encoding prompt (T5) ... done")

            del input_ids, attention_mask, prompt_embeds
            if HunyuanStorage.noUnload == False:
                HunyuanStorage.teT5 = None
            #end text_encode 2
        else:
            #256 is tokenizer max length from config; 2048 is transformer joint_attention_dim from its config
            positive_embeds_2    = torch.zeros((1, 256, 2048))
            positive_attention_2 = torch.zeros((1, 256))
            negative_embeds_2    = torch.zeros((1, 256, 2048))
            negative_attention_2 = torch.zeros((1, 256))

        #   pad embeds
        positive_embeds_2 = torch.nn.functional.pad(input=positive_embeds_2, pad=(0, 0, 0, 256-positive_embeds_2.size(1), 0, 0), mode='constant', value=0)
        negative_embeds_2 = torch.nn.functional.pad(input=negative_embeds_2, pad=(0, 0, 0, 256-negative_embeds_2.size(1), 0, 0), mode='constant', value=0)
        positive_attention_2 = torch.nn.functional.pad(input=positive_attention_2, pad=(0, 256-positive_attention_2.size(1), 0, 0), mode='constant', value=0)
        negative_attention_2 = torch.nn.functional.pad(input=negative_attention_2, pad=(0, 256-negative_attention_2.size(1), 0, 0), mode='constant', value=0)

        HunyuanStorage.positive_embeds_1    = positive_embeds_1.to(torch.float16)
        HunyuanStorage.positive_attention_1 = positive_attention_1.to(torch.float16)
        HunyuanStorage.negative_embeds_1    = negative_embeds_1.to(torch.float16)
        HunyuanStorage.negative_attention_1 = negative_attention_1.to(torch.float16)
        HunyuanStorage.positive_embeds_2    = positive_embeds_2.to(torch.float16)
        HunyuanStorage.positive_attention_2 = positive_attention_2.to(torch.float16)
        HunyuanStorage.negative_embeds_2    = negative_embeds_2.to(torch.float16)
        HunyuanStorage.negative_attention_2 = negative_attention_2.to(torch.float16)

        del positive_embeds_1, negative_embeds_1, positive_attention_1, negative_attention_1
        del positive_embeds_2, negative_embeds_2, positive_attention_2, negative_attention_2

        HunyuanStorage.lastPrompt = combined_positive
        HunyuanStorage.lastNegative = combined_negative

    gc.collect()
    torch.cuda.empty_cache()

    ####    set up pipe, VAE
    if HunyuanStorage.pipe == None:
        HunyuanStorage.pipe = pipeline.HunyuanDiTPipeline_DoE_combined.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            local_files_only=False, cache_dir=".//models//diffusers//",
            transformer=None,
            feature_extractor=None,
            torch_dtype=torch.float16,
            tokenizer=None,
            text_encoder=None,
            tokenizer_2=None,
            text_encoder_2=None,
            use_safetensors=True,
            controlnet=None,
        )

    ####    transformer
    source = "Tencent-Hunyuan/" + model
    if HunyuanStorage.lastTR != source:
        print ("Hunyuan: loading transformer ...", end="\r", flush=True)
        HunyuanStorage.pipe.transformer = HunyuanDiT2DModel.from_pretrained(
            source,
            local_files_only=False, cache_dir=".//models//diffusers//",
            subfolder='transformer',
            torch_dtype=torch.float16,
        )
        HunyuanStorage.lastTR = source

    if HunyuanStorage.lastControlNet != useControlNet and useControlNet:
        HunyuanStorage.pipe.controlnet=HunyuanDiT2DControlNetModel.from_pretrained(
            useControlNet, cache_dir=".//models//diffusers//", 
            low_cpu_mem_usage=False, device_map=None,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float16)
        HunyuanStorage.lastControlNet = useControlNet

    HunyuanStorage.pipe.enable_model_cpu_offload()
#    HunyuanStorage.pipe.vae.enable_slicing()

    #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
    shape = (
        num_images,
        HunyuanStorage.pipe.transformer.config.in_channels,
        int(height) // HunyuanStorage.pipe.vae_scale_factor,
        int(width) // HunyuanStorage.pipe.vae_scale_factor,
    )

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda').to(torch.float16)

    if HunyuanStorage.sharpNoise:
        minDim = 1 + 2*(min(latents.size(2), latents.size(3)) // 4)
        for b in range(len(latents)):
            blurred = TF.gaussian_blur(latents[b], minDim)
            latents[b] = 1.05*latents[b] - 0.05*blurred

    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

    #   colour the initial noise
    if HunyuanStorage.noiseRGBA[3] != 0.0:
        nr = HunyuanStorage.noiseRGBA[0] ** 0.5
        ng = HunyuanStorage.noiseRGBA[1] ** 0.5
        nb = HunyuanStorage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(numpy.full((8,8), (nr), dtype=numpy.float32))
        imageG = torch.tensor(numpy.full((8,8), (ng), dtype=numpy.float32))
        imageB = torch.tensor(numpy.full((8,8), (nb), dtype=numpy.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = HunyuanStorage.pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = HunyuanStorage.pipe.vae.encode(image).latent_dist.sample(generator)
        image_latents *= HunyuanStorage.pipe.vae.config.scaling_factor * HunyuanStorage.pipe.scheduler.init_noise_sigma
        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

        for b in range(len(latents)):
            for c in range(4):
                latents[b][c] -= latents[b][c].mean()

#        latents += image_latents * HunyuanStorage.noiseRGBA[3]
        torch.lerp (latents, image_latents, HunyuanStorage.noiseRGBA[3], out=latents)

        del imageR, imageG, imageB, image, image_latents
    #   end: colour the initial noise



    #   load LoRA; don't abort on failure, carry on without
    if HunyuanStorage.lora and HunyuanStorage.lora != "(None)" and HunyuanStorage.lora_scale != 0.0:
        lorafile = ".//models//diffusers//HunyuanLora//" + HunyuanStorage.lora + ".safetensors"

        try:
            from safetensors import safe_open
            lora_state_dict = {}
            with safe_open(lorafile, framework="pt", device=0) as f:
                for k in f.keys():
                    lora_state_dict[k] = f.get_tensor(k)
                    
            if "base_model.model.blocks.0.attn1.Wqkv.lora_A.weight" in lora_state_dict:
                #   needs converting from Tencent HY Lora
                transformer_state_dict = HunyuanStorage.pipe.transformer.state_dict()
                transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=HunyuanStorage.lora_scale)
                HunyuanStorage.pipe.transformer.load_state_dict(transformer_state_dict)
                HunyuanStorage.loadedLora = True
            elif "lora_unet_blocks_0_attn1_Wqkv.alpha" in lora_state_dict:
                #   needs converting from safetensors lora
                print ("Hunyuan: LoRA failed: unsupported type: " + lorafile)
            else:
                try:
                    #   already converted
                    HunyuanStorage.pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=HunyuanStorage.lora)
                    HunyuanStorage.loadedLora = True
                except:
                    # incompatible
                    print ("Hunyuan: LoRA failed: unsupported/incompatible type: " + lorafile)
            del lora_state_dict
        except:
            print ("Hunyuan: LoRA failed: file not found: " + lorafile)
    #   end: load LoRA

    schedulerConfig = dict(HunyuanStorage.pipe.scheduler.config)
    if "HunyuanDiT-v1.2-Diffusers" in model:
        schedulerConfig['beta_end'] = 0.018
    schedulerConfig['use_karras_sigmas'] = HunyuanStorage.karras
    if HunyuanStorage.zeroSNR:
        schedulerConfig['timestep_spacing'] = 'trailing'
        schedulerConfig['rescale_betas_zero_snr'] = True
#    schedulerConfig['lower_order_final'] = True
    schedulerConfig.pop('algorithm_type', None) 

    if scheduler == 'DDPM':
        HunyuanStorage.pipe.scheduler = DDPMScheduler.from_config(schedulerConfig)
    elif scheduler == 'DEIS':
        HunyuanStorage.pipe.scheduler = DEISMultistepScheduler.from_config(schedulerConfig)
    elif scheduler == 'DPM++ 2M':
        HunyuanStorage.pipe.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif scheduler == "DPM++ 2M SDE":
        schedulerConfig['algorithm_type'] = 'sde-dpmsolver++'
        HunyuanStorage.pipe.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif scheduler == 'DPM':
        HunyuanStorage.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(schedulerConfig)
    elif scheduler == 'DPM SDE':
        HunyuanStorage.pipe.scheduler = DPMSolverSDEScheduler.from_config(schedulerConfig)
    elif scheduler == 'Euler':
        HunyuanStorage.pipe.scheduler = EulerDiscreteScheduler.from_config(schedulerConfig)
    elif scheduler == 'Euler A':
        HunyuanStorage.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(schedulerConfig)
    elif scheduler == "SA-solver":
        schedulerConfig['algorithm_type'] = 'data_prediction'
        HunyuanStorage.pipe.scheduler = SASolverScheduler.from_config(schedulerConfig)
    elif scheduler == 'UniPC':
        HunyuanStorage.pipe.scheduler = UniPCMultistepScheduler.from_config(schedulerConfig)
    else:
        HunyuanStorage.pipe.scheduler = DDPMScheduler.from_config(schedulerConfig)


#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    HunyuanStorage.pipe.transformer.to(memory_format=torch.channels_last)
    HunyuanStorage.pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = HunyuanStorage.pipe(
            image                           = i2iSource,
            strength                        = i2iDenoise,
            mask_image                      = maskSource,
            mask_cutoff                     = maskCutOff,

            num_inference_steps             = num_steps,
            num_images_per_prompt           = num_images,
            height                          = height,
            width                           = width,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            guidance_cutoff                 = guidance_cutoff,
            centre_latents                  = HunyuanStorage.centreLatents,
            output_type                     = "latents",
            use_resolution_binning          = False,
            generator                       = generator,
            latents                         = latents,

            prompt_embeds                   = HunyuanStorage.positive_embeds_1,
            negative_prompt_embeds          = HunyuanStorage.negative_embeds_1,
            prompt_attention_mask           = HunyuanStorage.positive_attention_1,
            negative_prompt_attention_mask  = HunyuanStorage.negative_attention_1,
            prompt_embeds_2                 = HunyuanStorage.positive_embeds_2,
            negative_prompt_embeds_2        = HunyuanStorage.negative_embeds_2,
            prompt_attention_mask_2         = HunyuanStorage.positive_attention_2,
            negative_prompt_attention_mask_2= HunyuanStorage.negative_attention_2,

            control_image                   = controlNetImage, 
            controlnet_conditioning_scale   = controlNetStrength,  
            control_guidance_start          = controlNetStart,
            control_guidance_end            = controlNetEnd,

#            cross_attention_kwargs          = {"scale": HunyuanStorage.lora_scale }    #   currently does nothing - HYDiT forward pass doesn't take this input
        )

    if HunyuanStorage.noUnload:
        if HunyuanStorage.loadedLora == True:
            if HunyuanStorage.transformerStateDict: #how big? better to just clear transformer?
                HunyuanStorage.pipe.transformer.load_state_dict(HunyuanStorage.transformerStateDict)
                HunyuanStorage.transformerStateDict = None
            else:
                HunyuanStorage.pipe.unload_lora_weights()
            HunyuanStorage.loadedLora = False
    else:
        #   cannot delete pipe here as still need VAE
        HunyuanStorage.pipe.transformer = None
        HunyuanStorage.pipe.controlnet = None
        HunyuanStorage.lastTR = None
        HunyuanStorage.lastControlNet = None

    del generator, latents, controlNetImage

    gc.collect()
    torch.cuda.empty_cache()

    if HunyuanStorage.lora != "(None)" and HunyuanStorage.lora_scale != 0.0:
        loraSettings = HunyuanStorage.lora + f" ({HunyuanStorage.lora_scale})"
    else:
        loraSettings = None
    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}, step range: {controlNetStart}-{controlNetEnd}"

    result = []
    total = len(output)
    for i in range (total):
        print (f'Hunyuan: VAE: {i+1} of {total}', end='\r', flush=True)
        info=create_infotext(
            model, 
            combined_positive, combined_negative,
            guidance_scale, guidance_rescale, guidance_cutoff, num_steps, 
            fixed_seed + i, scheduler,
            width, height, 
            loraSettings, 
            useControlNet)

        latent = (output[i:i+1]) / HunyuanStorage.pipe.vae.config.scaling_factor
        image = HunyuanStorage.pipe.vae.decode(latent, return_dict=False)[0]
        image = HunyuanStorage.pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])[0]

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed + i,
            combined_positive,
            opts.samples_format,
            info
        )
    print ('Hunyuan: VAE: done  ')

#    if not HunyuanStorage.noUnload:
#       HunyuanStorage.pipe = None  #   contains scheduler, VAE

    del output
    gc.collect()
    torch.cuda.empty_cache()

    HunyuanStorage.locked = False
    return gradio.Button.update(value='Generate', variant='primary', interactive=True), gradio.Button.update(interactive=True), result


def on_ui_tabs():
    if HunyuanStorage.ModuleReload:
        reload (pipeline)

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
        return gradio.Dropdown.update(choices=loras)
   
    def getGalleryIndex (evt: gradio.SelectData):
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
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.captionToPrompt])


    #   these are volatile state, should not be changed during generation
    def toggleKarras ():
        if not HunyuanStorage.locked:
            HunyuanStorage.karras ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.karras],
                                value=['\U0001D542', '\U0001D40A'][HunyuanStorage.karras])
    def toggleT5 ():
        if not HunyuanStorage.locked:
            HunyuanStorage.lastPrompt = None
            HunyuanStorage.lastNegative = None
            HunyuanStorage.useT5 ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.useT5])
    def toggleAS ():
        if not HunyuanStorage.locked:
            HunyuanStorage.i2iAllSteps ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.i2iAllSteps])
    def toggleCL ():
        if not HunyuanStorage.locked:
            HunyuanStorage.centreLatents ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.centreLatents],
                                value=['\u29BE', '\u29BF'][HunyuanStorage.centreLatents])
    def toggleZSNR ():
        if not HunyuanStorage.locked:
            HunyuanStorage.zeroSNR ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.zeroSNR])

    def toggleSP ():
        if not HunyuanStorage.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        if HunyuanStorage.locked:
            return gradio.Button.update(variant='secondary'), prompt
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                cache_dir='.//models//diffusers//',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')

        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result


    resolutionList = [
        (1280, 960),    (1280, 768),    (1152, 864),    (1024, 768),
        (1280, 1280),   (1024, 1024),
        (768, 1024),    (864, 1152),    (768, 1280),    (960, 1280)
    ]

    def updateWH (idx, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        if idx < len(resolutionList):
            return None, resolutionList[idx][0], resolutionList[idx][1]
        return None, w, h

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
                                                         torch_dtype=torch.float16, 
                                                         cache_dir=".//models//diffusers//", 
                                                         trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  cache_dir=".//models//diffusers//", 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image, return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
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
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

        del model, processor

        if HunyuanStorage.captionToPrompt:
            return result
        else:
            return originalPrompt

    def toggleGenerate (R, G, B, A, lora, scale):
        HunyuanStorage.noiseRGBA = [R, G, B, A]
        HunyuanStorage.lora = lora
        HunyuanStorage.lora_scale = scale# if lora != "(None)" else 1.0
        HunyuanStorage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)

    schedulerList = ["DDPM", "DEIS", "DPM++ 2M", "DPM++ 2M SDE", "DPM", "DPM SDE", "SA-solver", "UniPC", ]

    def parsePrompt (positive, negative, width, height, seed, scheduler, steps, CFG, CFGrescale, CFGcutoff, nr, ng, nb, ns, loraName, loraScale):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = int(size[0])
                            height = int(size[1])
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Sampler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Scheduler:":
                            sched = ' '.join(pairs[1:])
                            if sched in schedulerList:
                                scheduler = sched
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                CFG = float(pairs[2])
                        case "CFG:":
                            CFG = float(pairs[1])
                            if len(pairs) == 4:
                                CFGrescale = float(pairs[2].strip('\(\)'))
                                CFGcutoff = float(pairs[3].strip('\[\]'))
                        case "width:":
                            width = float(pairs[1])
                        case "height:":
                            height = float(pairs[1])
                        case "LoRA:":
                            if len(pairs) == 3:
                                loraName = pairs[1]
                                loraScale = float(pairs[2].strip('\(\)'))

        return positive, negative, width, height, seed, scheduler, steps, CFG, CFGrescale, CFGcutoff, nr, ng, nb, ns, loraName, loraScale

    def toggleSharp ():
        if not HunyuanStorage.locked:
            HunyuanStorage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][HunyuanStorage.sharpNoise],
                                variant=['secondary', 'primary'][HunyuanStorage.sharpNoise])

    def maskFromImage (image):
        if image:
            return image, 'drawn'
        else:
            return None, 'none'

    def toggleNU ():
        if not HunyuanStorage.locked:
            HunyuanStorage.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][HunyuanStorage.noUnload])
    def unloadM ():
        if not HunyuanStorage.locked:
            HunyuanStorage.teCLIP = None
            HunyuanStorage.teT5 = None
            HunyuanStorage.pipe = None
            HunyuanStorage.lastTR = None
            HunyuanStorage.lastControlNet = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')

    with gradio.Blocks() as hunyuandit_block:
        with ResizeHandleRow():
            with gradio.Column():
                with gradio.Row():
                    model = gradio.Dropdown(['HunyuanDiT-v1.2-Diffusers-Distilled',
                                             'HunyuanDiT-v1.2-Diffusers',
                                             'HunyuanDiT-v1.1-Diffusers-Distilled',
                                             'HunyuanDiT-v1.1-Diffusers',
                                             'HunyuanDiT-Diffusers-Distilled',
                                             'HunyuanDiT-Diffusers',
                                            ], label='Model', value='HunyuanDiT-v1.2-Diffusers-Distilled', type='value')

                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='zero out negative embeds')
                    T5 = ToolButton(value="T5", variant='primary', tooltip="use T5 text encoder")
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
                    CL = ToolButton(value='\u29BE', variant='secondary', tooltip='centre latents to mean')
                    zsnr = ToolButton(value='zsnr', variant='secondary', tooltip='zero SNR')
                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=1.01)
                    scheduler = gradio.Dropdown(schedulerList,
                        label='Sampler', value="SA-solver", type='value', scale=0)

                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', placeholder='', lines=1.01)
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_width")
                    swapper = ToolButton(value="\U000021C4")
                    height = gradio.Slider(label='Height', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_height")
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList],
                                        label='Quickset', type='index', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=4)
                    guidance_rescale = gradio.Slider(label='rescale CFG', minimum=0, maximum=1, step=0.01, value=0)
                    CFGcutoff = gradio.Slider(label='CFG cutoff after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0, precision=0.01, scale=1)
                with gradio.Row():
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gradio.Row(equal_height=True):
                    lora = gradio.Dropdown([x for x in loras], label='LoRA (place in models/diffusers/HunyuanLora)', value="(None)", type='value', multiselect=False, scale=1)
                    refresh = ToolButton(value='\U0001f504')
                    scale = gradio.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='ControlNet', open=False):
                    with gradio.Row():
                        CNSource = gradio.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gradio.Column():
                            CNMethod = gradio.Dropdown(['(None)', 'canny', 'depth', 'pose'], label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gradio.Slider(label='Strength', minimum=0.00, maximum=2.0, step=0.01, value=0.8)
                            CNStart = gradio.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gradio.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gradio.Accordion(label='image to image', open=False):
                    with gradio.Row():
                        i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        maskSource = gradio.Image(label='source mask', sources=['upload'], type='pil', interactive=True, show_download_button=False, tool='sketch', image_mode='RGB', brush_color='#F0F0F0')#opts.img2img_inpaint_mask_brush_color)
                    with gradio.Row():
                        with gradio.Column():
                            with gradio.Row():
                                i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                AS = ToolButton(value='AS')
                            with gradio.Row():
                                i2iFromGallery = gradio.Button(value='Get gallery image')
                                i2iSetWH = gradio.Button(value='Set size from image')
                            with gradio.Row():
                                i2iCaption = gradio.Button(value='Caption image (Florence-2)', scale=6)
                                toPrompt = ToolButton(value='P', variant='secondary')

                        with gradio.Column():
                            maskType = gradio.Dropdown(['none', 'image', 'drawn'], value='none', label='Mask', type='index')
                            maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=25, step=1, value=0)
                            maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                            maskCopy = gradio.Button(value='use i2i source as template')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if HunyuanStorage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                ctrls = [model, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, CFGcutoff, steps, sampling_seed, batch_size, scheduler, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCut, CNMethod, CNSource, CNStrength, CNStart, CNEnd]

                parseable = [positive_prompt, negative_prompt, width, height, sampling_seed, scheduler, steps, guidance_scale, guidance_rescale, CFGcutoff, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh",
                                            show_label=False, visible=True, object_fit='none', columns=1, preview=True)
#   gallery movement buttons don't work, others do
#   caption not displaying linebreaks, alt text does

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=positive_prompt,
                        source_image_component=output_gallery,
                    ))

        noUnload.click(toggleNU, inputs=[], outputs=noUnload)
        unloadModels.click(unloadM, inputs=[], outputs=[], show_progress=True)
        SP.click(toggleSP, inputs=[], outputs=SP)
        SP.click(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        sharpNoise.click(toggleSharp, inputs=[], outputs=sharpNoise)
        maskCopy.click(fn=maskFromImage, inputs=[i2iSource], outputs=[maskSource, maskType])

        parse.click(parsePrompt, inputs=parseable, outputs=parseable, show_progress=False)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        refresh.click(refreshLoRAs, inputs=[], outputs=[lora])
        karras.click(toggleKarras, inputs=[], outputs=karras)
        T5.click(toggleT5, inputs=[], outputs=T5)
        AS.click(toggleAS, inputs=[], outputs=AS)
        CL.click(toggleCL, inputs=[], outputs=CL)
        zsnr.click(toggleZSNR, inputs=[], outputs=zsnr)
        swapper.click(fn=None, _js="function(){switchWidthHeight('Hunyuan-DiT')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])
        i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])#outputs=[positive_prompt]
        toPrompt.click(toggleC2P, inputs=[], outputs=[toPrompt])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(predict, inputs=ctrls, outputs=[generate_button, SP, output_gallery])
        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button, SP])

    return [(hunyuandit_block, "Hunyuan-DiT", "hunyuan")]

script_callbacks.on_ui_tabs(on_ui_tabs)

