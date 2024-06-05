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
    pos_embeds = None
    pos_attention = None
    neg_embeds = None
    neg_attention = None
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
def create_infotext(positive_prompt, negative_prompt, guidance_scale, steps, seed, scheduler, width, height):
    karras = " : Karras" if HunyuanStorage.karras == True else ""
    generation_params = {
        "Size": f"{width}x{height}",
        "Seed": seed,
        "Scheduler": f"{scheduler}{karras}",
        "Steps": steps,
        "CFG": guidance_scale,
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None
    }


    prompt_text = f"Prompt: {positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    return f"Model: Hunyuan-DiT\n{prompt_text}{generation_params_text}"

def predict(positive_prompt, negative_prompt, width, height, guidance_scale, num_steps, sampling_seed, num_images, scheduler, style, *args):

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

#    useCachedEmbeds = False#(HunyuanStorage.lastPrompt == positive_prompt and HunyuanStorage.lastNegative == negative_prompt)

#    logging.set_verbosity(logging.ERROR)

    pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None, )

#    del pipe.vae            #stupid implementation needs transformer and vae during pipe init
#    del pipe.transformer


    (
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        device='cpu',
        dtype=torch.float16,
        num_images_per_prompt=num_images,
        do_classifier_free_guidance=True,
##            prompt_embeds=prompt_embeds,
##            negative_prompt_embeds=negative_prompt_embeds,
##            prompt_attention_mask=prompt_attention_mask,
##            negative_prompt_attention_mask=negative_prompt_attention_mask,
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
        device='cpu',
        dtype=torch.float16,
        num_images_per_prompt=num_images,
        do_classifier_free_guidance=True,
##            prompt_embeds=prompt_embeds_2,
##            negative_prompt_embeds=negative_prompt_embeds_2,
##            prompt_attention_mask=prompt_attention_mask_2,
##            negative_prompt_attention_mask=negative_prompt_attention_mask_2,
        max_sequence_length=256,
        text_encoder_index=1,
    )


    del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    pipe.text_encoder = None
    pipe.text_encoder_2 = None

    gc.collect()
    torch.cuda.empty_cache()

##    pipe = HunyuanDiTPipeline.from_pretrained(
##        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
##        requires_safety_checker=False,
##        safety_checker=None,
##        feature_extractor=None,
##        tokenizer=None,
##        text_encoder=None,
##        tokenizer_2=None,
##        text_encoder_2=None,)

    pipe.to('cuda')
    pipe.enable_model_cpu_offload()

    pipe.vae.enable_tiling(True)
#    pipe.enable_attention_slicing()

    with torch.no_grad():
        #   if using resolution_binning, must use adjusted width/height here (don't overwrite values)
        #   always generate the noise here
        generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]

    
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
        prompt_embeds=prompt_embeds.to('cuda').to(torch.float16),
        negative_prompt_embeds=negative_prompt_embeds.to('cuda').to(torch.float16),
        prompt_attention_mask=prompt_attention_mask.to('cuda').to(torch.float16),
        negative_prompt_attention_mask=negative_prompt_attention_mask.to('cuda').to(torch.float16),
        prompt_embeds_2=prompt_embeds_2.to('cuda').to(torch.float16),
        negative_prompt_embeds_2=negative_prompt_embeds_2.to('cuda').to(torch.float16),
        prompt_attention_mask_2=prompt_attention_mask_2.to('cuda').to(torch.float16),
        negative_prompt_attention_mask_2=negative_prompt_attention_mask_2.to('cuda').to(torch.float16),
        num_images_per_prompt=num_images,
        output_type="pil",
        generator=generator,
        use_resolution_binning=False,
    ).images

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


    result = []
    for image in output:
        info=create_infotext(
            positive_prompt, negative_prompt,
            guidance_scale, num_steps, 
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
                    positive_prompt = gr.Textbox(label='Prompt', placeholder='Enter a prompt here...', default='', lines=2)
                    scheduler = gr.Dropdown(["default",
                                             "DDPM",
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
                    negative_prompt = gr.Textbox(label='Negative', placeholder='', lines=2)
                    style = gr.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)
                with gr.Row():
                    width = gr.Slider(label='Width', minimum=512, maximum=1536, step=32, value=1024, elem_id="Hunyuan-DiT_width")
                    swapper = ToolButton(value="\U000021C5")
                    height = gr.Slider(label='Height', minimum=512, maximum=1536, step=32, value=1024, elem_id="Hunyuan-DiT_height")

                with gr.Row():
                    guidance_scale = gr.Slider(label='CFG', minimum=1, maximum=16, step=0.5, value=1, scale=2)
                    steps = gr.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                with gr.Row():
                    sampling_seed = gr.Number(label='Seed', value=-1, precision=0, scale=1)
                    random = ToolButton(value="\U0001f3b2\ufe0f")
                    reuseSeed = ToolButton(value="\u267b\ufe0f")
                    batch_size = gr.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                ctrls = [positive_prompt, negative_prompt, width, height, guidance_scale, steps, sampling_seed, batch_size, scheduler, style]

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

        output_gallery.select (fn=getGalleryIndex, inputs=[], outputs=[])

        generate_button.click(toggleGenerate, inputs=[], outputs=[generate_button])
        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery, generate_button])

    return [(hunyuandit_block, "Hunyuan-DiT", "hunyuan")]

script_callbacks.on_ui_tabs(on_ui_tabs)

