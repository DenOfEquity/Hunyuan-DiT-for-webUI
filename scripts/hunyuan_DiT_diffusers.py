import math
import torch
import gc
import json
import numpy as np

from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.rng import create_generator
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste
import gradio as gr

from PIL import Image

torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_mem_efficient_sdp(True)


import customStylesListHY as styles

class HunyuanStorage:
    lastSeed = -1
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
    useDistilled = False

from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

from diffusers import HunyuanDiTPipeline
from diffusers import AutoencoderKL, HunyuanDiT2DModel

from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler

from diffusers.utils.torch_utils import randn_tensor

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
def create_infotext(positive_prompt, negative_prompt, guidance_scale, guidance_rescale, steps, seed, scheduler, width, height):
    karras = " : Karras" if HunyuanStorage.karras == True else ""
    distilled = " : distilled" if HunyuanStorage.useDistilled == True else ""
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps": steps,
        "CFG": f"{guidance_scale}({guidance_rescale})",
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }


    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: Hunyuan-DiT{distilled}\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale,
            num_steps, sampling_seed, num_images, scheduler, style, i2iSource, i2iDenoise, *args):

    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = styles.styles_list[style][2] + negative_prompt

    if i2iSource == None:
        i2iDenoise = 1
    if i2iDenoise < (num_steps + 1) / 1000:
        i2iDenoise = (num_steps + 1) / 1000

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    HunyuanStorage.lastSeed = fixed_seed

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    #first: tokenize and text_encode
    useCachedEmbeds = (HunyuanStorage.lastPrompt == positive_prompt and HunyuanStorage.lastNegative == negative_prompt)
    if useCachedEmbeds:
        print ("Skipping text encoders and tokenizers.")
        #   nothing to do
    else:
        with torch.no_grad():
            #   tokenize 1
            tokenizer = BertTokenizer.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer',
                torch_dtype=torch.float16,
                )
            #   positive
            text_inputs = tokenizer(
                positive_prompt,
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
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            negative_text_input_ids = text_inputs.input_ids.to('cuda')
            negative_attention_1 = text_inputs.attention_mask.to('cuda')

            del tokenizer
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
            positive_embeds_1 = prompt_embeds[0]
            positive_attention_1 = positive_attention_1.repeat(num_images, 1)

            prompt_embeds = text_encoder(
                negative_text_input_ids,
                attention_mask=negative_attention_1,
            )
            negative_embeds_1 = prompt_embeds[0]
            negative_attention_1 = negative_attention_1.repeat(num_images, 1)

            del text_encoder
            #end text_encode 1

            gc.collect()
            torch.cuda.empty_cache()


            #   tokenize 2
            tokenizer = MT5Tokenizer.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-Diffusers",
                local_files_only=False, cache_dir=".//models//diffusers//",
                subfolder='tokenizer_2',
                torch_dtype=torch.float16,
                )
            #   positive
            text_inputs = tokenizer(
                positive_prompt,
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
                negative_prompt,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            negative_text_input_ids = text_inputs.input_ids.to('cuda')
            negative_attention_2 = text_inputs.attention_mask.to('cuda')

            del tokenizer
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
            positive_embeds_2 = prompt_embeds[0]
            positive_attention_2 = positive_attention_2.repeat(num_images, 1)

            prompt_embeds = text_encoder(
                negative_text_input_ids,
                attention_mask=negative_attention_2,
            )
            negative_embeds_2 = prompt_embeds[0]
            negative_attention_2 = negative_attention_2.repeat(num_images, 1)

            del text_encoder
            #end text_encode 2

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

        HunyuanStorage.lastPrompt = positive_prompt
        HunyuanStorage.lastNegative = negative_prompt

    gc.collect()
    torch.cuda.empty_cache()

    #second: transformer/VAE
    source = "Tencent-Hunyuan/HunyuanDiT-Diffusers-Distilled" if HunyuanStorage.useDistilled else "Tencent-Hunyuan/HunyuanDiT-Diffusers"

    transformer = HunyuanDiT2DModel.from_pretrained(
        source,
        local_files_only=False, cache_dir=".//models//diffusers//",
        subfolder='transformer',
        torch_dtype=torch.float16,
        )
 
    pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        local_files_only=False, cache_dir=".//models//diffusers//",
        transformer=transformer,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=torch.float16,
        tokenizer=None,
        text_encoder=None,
        tokenizer_2=None,
        text_encoder_2=None,
        use_safetensors=True,
        )
    pipe.to('cuda')
    pipe.enable_model_cpu_offload()
       
#    pipe.transformer.enable_forward_chunking(chunk_size=1, dim=1)      #>= 0.28.2 ?
    pipe.vae.enable_tiling(True)
#    pipe.enable_attention_slicing()

    with torch.no_grad():
        #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
        #   always generate the noise here
        generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
        shape = (
            num_images,
            pipe.transformer.config.in_channels,
            int(height) // pipe.vae_scale_factor,
            int(width) // pipe.vae_scale_factor,
        )

        i2i_latents = randn_tensor(shape, generator=generator, dtype=torch.float16).to('cuda').to(torch.float16)

        if i2iSource != None:
            i2iSource = i2iSource.resize((width, height))

            image = pipe.image_processor.preprocess(i2iSource).to('cuda').to(torch.float16)
            image_latents = pipe.vae.encode(image).latent_dist.sample(generator) * pipe.vae.config.scaling_factor * pipe.scheduler.init_noise_sigma
            image_latents = image_latents.repeat(num_images, 1, 1, 1)

            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
            ts = torch.tensor([int(1000 * i2iDenoise) - 1], device='cpu')
            ts = ts[:1].repeat(num_images)

            i2i_latents = pipe.scheduler.add_noise(image_latents, i2i_latents, ts)

            del image, image_latents, i2iSource
    
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

    pipe.scheduler.config.num_train_timesteps = int(1000 * i2iDenoise)
    pipe.scheduler.config.use_karras_sigmas = HunyuanStorage.karras

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas

    output = pipe(
        prompt=None,
        negative_prompt=None, 
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
        latents=i2i_latents,
    ).images

    del pipe, generator, i2i_latents

    gc.collect()
    torch.cuda.empty_cache()


    result = []
    for image in output:
        info=create_infotext(
            positive_prompt, negative_prompt,
            guidance_scale, guidance_rescale, num_steps, 
            fixed_seed, scheduler,
            width, height, )

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed,
            positive_prompt,
            opts.samples_format,
            info
        )
        fixed_seed += 1

    del output
    gc.collect()
    torch.cuda.empty_cache()

    return result, gr.Button.update(value='Generate', variant='primary', interactive=True)


def on_ui_tabs():
   
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

    def toggleKarras ():
        if HunyuanStorage.karras == False:
            HunyuanStorage.karras = True
            return gr.Button.update(value='\U0001D40A', variant='primary')
        else:
            HunyuanStorage.karras = False
            return gr.Button.update(value='\U0001D542', variant='secondary')
    def toggleDistilled ():
        if HunyuanStorage.useDistilled == False:
            HunyuanStorage.useDistilled = True
            return gr.Button.update(value='\U0001D403', variant='primary')
        else:
            HunyuanStorage.useDistilled = False
            return gr.Button.update(value='\U0001D53B', variant='secondary')


    def toggleGenerate ():
        return gr.Button.update(value='...', variant='secondary', interactive=False)

    with gr.Blocks() as hunyuandit_block:
        with ResizeHandleRow():
            with gr.Column():
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
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
                    distilled = ToolButton(value="\U0001D53B", variant='secondary', tooltip="use distilled model")


                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=1.1)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=768, maximum=1280, step=32, value=1024, elem_id="Hunyuan-DiT_height")

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.5, value=1, scale=2)
                    guidance_rescale = gr.Slider(label='rescale CFG', minimum=0, maximum=1, step=0.01, value=0, scale=2)
                with gr.Row():
                    steps = gr.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gr.Accordion(label='image to image', open=False):
                    with gr.Row():
                        i2iSource = gr.Image(label='image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gr.Column():
                            i2iDenoise = gr.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            i2iSetWH = gr.Button(value='Set safe Width / Height from image')
                            i2iFromGallery = gr.Button(value='Get image from gallery')

                ctrls = [positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, steps, sampling_seed,
                         batch_size, scheduler, style, i2iSource, i2iDenoise]

            with gr.Column():
                generate_button = gr.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gr.Gallery(label='Output', height=shared.opts.gallery_height or None,
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


        karras.click(toggleKarras, inputs=[], outputs=karras)
        distilled.click(toggleDistilled, inputs=[], outputs=distilled)
        swapper.click(fn=None, _js="function(){switchWidthHeight('Hunyuan-DiT')}", inputs=None, outputs=None, show_progress=False)
        random.click(randomSeed, inputs=[], outputs=sampling_seed, show_progress=False)
        reuseSeed.click(reuseLastSeed, inputs=[], outputs=sampling_seed, show_progress=False)

        i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
        i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery], outputs=[i2iSource])

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(toggleGenerate, inputs=[], outputs=[generate_button])
        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery, generate_button])

    return [(hunyuandit_block, "Hunyuan-DiT", "hunyuan")]

script_callbacks.on_ui_tabs(on_ui_tabs)

