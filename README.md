## This repo based on diffusers lib and TheLastBen code

This script uses dreambooth technique, but with posibillity to train style via captions 
for all images (not just single concept). It save network as Lora, and may be merged in model back.
In general, it's cheaper then full-fine-tuning but strange and may not work. Also,
Dreambooth like to overfit: https://huggingface.co/blog/dreambooth
Script learn only unet layer for now. 


P.S.: I'm not python dev, so pls don't ask me how to fix strange errors. Thank you!

Unsplash_dev - service for download liked images from unsplash. I'm golang dev so feel free to ask about strange errors (but with golang i usually did't see mystical errors)

# TheLastBen
https://github.com/TheLastBen/diffusers

# Doc
https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md

# Colab
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb

## Install

# Clone
git clone https://github.com/recoilme/train && cd train

# Check the GPU
nvidia-smi

# Install dependencies.
pip install xformers (tons of downloads)

pip install -r requirements.txt

pip install bitsandbytes

accelerate config default


# Diffusers
pip install git+https://github.com/huggingface/diffusers.git 

# huggingface 
Make sure to log into your Hugging Face account and pass your access token so that we can push the trained checkpoints to the Hugging Face

huggingface-cli login

## Train

python3 train_sdxl.py --max_train_steps=300 --save_n_steps=100


## Tested a little on
ubuntu 20.04 + python 3.8 + 3090 Gpu

resulting lora on 3 img/300 steps
