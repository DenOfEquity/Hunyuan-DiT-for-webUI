import math
import torch
import gc
import json
import numpy as np
#
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


import customStylesList as styles

class HunyuanStorage:
    lastSeed = -1
    lastPrompt = None
    lastNegative = None
    prompt_embeds = None
    negative_prompt_embeds = None
    prompt_attention_mask = None
    negative_prompt_attention_mask = None
    prompt_embeds_2 = None
    negative_prompt_embeds_2 = None
    prompt_attention_mask_2 = None
    negative_prompt_attention_mask_2 = None
    karras = False

from transformers import BertModel, BertTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel

from diffusers import HunyuanDiTPipeline
from diffusers import AutoencoderKL


from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler
#from peft import PeftModel, PeftConfig

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

    return f"Model: Hunyuan-DiT\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, num_steps, sampling_seed, num_images, scheduler, style, i2iSource, i2iDenoise, *args):

    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = styles.styles_list[style][2] + negative_prompt

#    from diffusers.utils import logging
 #   logging.set_verbosity(logging.WARN)       #   download information is useful

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(sampling_seed)
    HunyuanStorage.lastSeed = fixed_seed

#    algorithm_type = args.algorithm
#    beta_schedule = args.beta_schedule
#    use_lu_lambdas = args.use_lu_lambdas

    useCachedEmbeds = (HunyuanStorage.lastPrompt == positive_prompt and HunyuanStorage.lastNegative == negative_prompt)

#    logging.set_verbosity(logging.ERROR)

    if useCachedEmbeds:
        print ("Skipping text encoders and tokenizers.")
        pipe = HunyuanDiTPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
            device='cuda',
            torch_dtype=torch.float16,
            tokenizer=None,
            text_encoder=None,
            tokenizer_2=None,
            text_encoder_2=None
            )
        pipe.enable_model_cpu_offload()
        prompt_embeds = HunyuanStorage.prompt_embeds.to('cuda')
        negative_prompt_embeds = HunyuanStorage.negative_prompt_embeds.to('cuda')
        prompt_attention_mask = HunyuanStorage.prompt_attention_mask.to('cuda')
        negative_prompt_attention_mask = HunyuanStorage.negative_prompt_attention_mask.to('cuda')
        prompt_embeds_2 = HunyuanStorage.prompt_embeds_2.to('cuda')
        negative_prompt_embeds_2 = HunyuanStorage.negative_prompt_embeds_2.to('cuda')
        prompt_attention_mask_2 = HunyuanStorage.prompt_attention_mask_2.to('cuda')
        negative_prompt_attention_mask_2 = HunyuanStorage.negative_prompt_attention_mask_2.to('cuda')
       
    else:
        pipe = HunyuanDiTPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-Diffusers",
            requires_safety_checker=False,
            safety_checker=None,
            feature_extractor=None,
            device='cuda',
            torch_dtype=torch.float16,
            )
        pipe.enable_model_cpu_offload()
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_attention_mask,
                negative_prompt_attention_mask,
            ) = pipe.encode_prompt(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                device='cuda',
                dtype=torch.float16,
                num_images_per_prompt=num_images,
                do_classifier_free_guidance=True,
                max_sequence_length=77,
                text_encoder_index=0,
            )
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_attention_mask_2,
                negative_prompt_attention_mask_2,
            ) = pipe.encode_prompt(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                device='cuda',
                dtype=torch.float16,
                num_images_per_prompt=num_images,
                do_classifier_free_guidance=True,
                max_sequence_length=256,
                text_encoder_index=1,
            )
        HunyuanStorage.prompt_embeds = prompt_embeds.to('cpu')
        HunyuanStorage.negative_prompt_embeds = negative_prompt_embeds.to('cpu')
        HunyuanStorage.prompt_attention_mask = prompt_attention_mask.to('cpu')
        HunyuanStorage.negative_prompt_attention_mask = negative_prompt_attention_mask.to('cpu')
        HunyuanStorage.prompt_embeds_2 = prompt_embeds_2.to('cpu')
        HunyuanStorage.negative_prompt_embeds_2 = negative_prompt_embeds_2.to('cpu')
        HunyuanStorage.prompt_attention_mask_2 = prompt_attention_mask_2.to('cpu')
        HunyuanStorage.negative_prompt_attention_mask_2 = negative_prompt_attention_mask_2.to('cpu')
        HunyuanStorage.lastPrompt = positive_prompt
        HunyuanStorage.lastNegative = negative_prompt

        del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2
        pipe.tokenizer = None
        pipe.tokenizer_2 = None
        pipe.text_encoder = None
        pipe.text_encoder_2 = None

        gc.collect()
        torch.cuda.empty_cache()

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
#   else uses default set by model

    pipe.scheduler.config.num_train_timesteps = int(1000 * i2iDenoise)
    pipe.scheduler.config.use_karras_sigmas = HunyuanStorage.karras

##    pipe.scheduler.beta_schedule  = beta_schedule
##    pipe.scheduler.use_lu_lambdas = use_lu_lambdas

    output = pipe(
        prompt=None,#positive_prompt,
        negative_prompt=None,#negative_prompt, 
        num_inference_steps=num_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
#        guidance_rescale=guidance_rescale,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        prompt_embeds_2=prompt_embeds_2,
        negative_prompt_embeds_2=negative_prompt_embeds_2,
        prompt_attention_mask_2=prompt_attention_mask_2,
        negative_prompt_attention_mask_2=negative_prompt_attention_mask_2,
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


                with gr.Row():
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=1.1)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=512, maximum=1536, step=32, value=1024, elem_id="Hunyuan-DiT_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=512, maximum=1536, step=32, value=1024, elem_id="Hunyuan-DiT_height")

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.5, value=1, scale=2)
                    guidance_rescale = gr.Slider(label='rescale CFG', minimum=0, maximum=16, step=1, value=0, scale=2)
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

