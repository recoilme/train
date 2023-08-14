## This repo based on diffusers lib and TheLastBen code

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

## train

python3 train_sdxl.py --max_train_steps=300

## Tested
tested on ubuntu 20.04 + python 3.8 + 3090 Gpu