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