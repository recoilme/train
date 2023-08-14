from diffusers import DiffusionPipeline
import torch

from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
)
pipe.load_lora_weights("sayakpaul/lora-trained-xl-colab")

_ = pipe.to("cuda")

prompt = "a photo of sks dog in a bucket"

image = pipe(prompt=prompt, num_inference_steps=25).images[0]
image.save("sks_dog.png")