## Hunyuan-DiT for webui ##
### Forge tested, probably A1111 too ###
I don't think there is anything Forge specific here.
### works for me <sup>TM</sup> on 8Gb VRAM, 16Gb RAM (GTX1070) ###

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.

---
### screenshot ###
current UI

![](screenshot.png "UI screenshot")


---
### downloads models on first run - ~13.4GB ###
### needs updated *diffusers 0.28.1* ###

Easiest way to ensure necessary versions are installed is to edit **requirements.text** and **requirements_versions.txt** in the webUI folder.
```
diffusers>=0.28.1
```

---
#### 07/06/2024 ####
* !! don't apply i2i denoise strength when not doing i2i, late night me forgot to copy that over from the PixArt implementation
* enabled guidance rescale for testing

#### 05/06/2024 ####
* reduced VRAM, no longer flirting with shared memory
* caching of prompt embeds to avoid text encoder processing if prompt and negative not changed
* img2img, same method as used with PixArt

image2image progression with a nice denoise

![](i2i.png "image2image sequence")

---

Generating with 8GB VRAM is possible. Using CFG 1 saves some VRAM and is considerably faster, but still slower than equivalent resolutions with sdXL or PixArt. Using small resolutions (768x768) seems to give very poor/broken results. Resolution binning is NOT enabled (width/height would be automatically adjusted to 'supported' values) as this seems to cause issues along borders.

---

### example ###
photograph of a kintsugi bowl of steaming dumplings on a rustic wooden table

![](example.png "896x1152, SA-solver, 20 steps, CFG 1")
