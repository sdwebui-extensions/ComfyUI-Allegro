## ComfyUI-MagicDance
ComfyUI supports over rhymes-ai/Allegro, which use text prompt to generate short video in relatively high quality, especially comparing to other open source solutions available for now.

## Installation

_Assuming that you are under your ComfyUI root directory_

git clone https://github.com/bombax-xiaoice/ComfyUI-Allegro custom_nodes/ComfyUI-Allegro

pip install -r custom_nodes/ComfyUI-Allegro/requirements.txt

_You can download the model file from huggingface or its mirror site beforehand, or just wait for the first run of (Down)Load Allegro Model to download it_

git lfs clone https://huggingface.co/rhymes-ai/Allegro custom_nodes/ComfyUI-Allegro/models

git lfs clone https://hf-mirror.com/rhymes-ai/Allegro custom_nodes/ComfyUI-Allegro/models

## Example Workflow
Drag the following image into comfyui, or click Load for custom_nodes/ComfyUI-Allegro/allegro-comfy-example.json

![](allegro-comfy-example.png)

Results run under comfy based on poses and image provided by MagicDance

<video src="allegro-comfy-example.mp4" width="1280" height="720" controls></video>

## Tips
Only verified that frame=88,width=1280,height=720 is working. Tried 24 frames and the result looks like random mosaics. Also tried width=560, where noisy bars show up along both left and right edges.

In (Down)Load Allegro Model, only provide the model path and leave others blank, unless you want to use alternative text encoder or vae models not provided by https://huggingface.co/rhymes-ai/Allegro

A default negative prompt will be used if you leave it blank in Allegro Text Encoder. A static template will also apply to the positive prompt.

Can skip Allegro Sampler's input latents to use frames/width/height to initialize it randomly. Otherwise, the batch size of input latents must be set as 1/4 of the desirable frames count.

Verified to work on a single NVidia RTX 3070 card with 8G graphics memory, where __low_vram_mode__ is turned on to load 32 layers of unet transformer block one by one into gpu vram, VAE decoder is also loaded seperately, and text encoder fall over to cpu. 

If you have enough graphics memory. You can try use --highvram on comfy start, where the entire pipeline is loaded into GPU directly to spare unnecessary conversion between CPU and GPU.

It is recommend to choose a preview method (inside comfy Manager), so that you can see the progress of step during the long run.

Email zhouli@xiaobing.ai for further question or discussion.