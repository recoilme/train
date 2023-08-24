## This repo based on diffusers lib and TheLastBen code

This script uses dreambooth technique, but with posibillity to train style via captions 
for all images (not just single concept). It save network as Lora, and may be merged in model back.
In general, it's cheaper then full-fine-tuning but strange and may not work. Also,
Dreambooth like to overfit: https://huggingface.co/blog/dreambooth
Script learn only unet layer for now. 


P.S.: I'm not python dev, so pls don't ask me how to fix strange errors. Thank you!

Unsplash_dev - service for download liked images from unsplash. I'm golang dev so feel free to ask about strange errors (but with golang i usually did't see mystical errors)

## literature
 - Formulas for training https://github.com/d8ahazard/sd_dreambooth_extension/discussions/547
 
 - TheLastBen https://github.com/TheLastBen/diffusers

 -  Doc https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md

 - Colab https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/SDXL_DreamBooth_LoRA_.ipynb

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

## It work a little, but badly! Loss is very unstable! Switched on Kohya scripts for full finetune

resulting lora on 3 img/300 steps
![lora](https://github.com/recoilme/train/assets/417177/d94a8f16-7071-4f15-a348-0980bac8bbf7)

## Kohya train params for 3090 with 24 RAM

```
accelerate launch  --num_cpu_threads_per_process 4 sdxl_train.py --resolution=1024,1024  --output_dir=mdl --output_name=tst2 --save_precision=fp16 --max_train_steps=5730 --mixed_precision=bf16 --logging_dir=mdl/log  --sample_every_n_steps=573 --seed=1234 --sample_prompts=prompt.txt --save_model_as=safetensors --train_data_dir=mdl/img --pretrained_model_name_or_path=/home/user/stable-diffusion-webui/models/Stable-diffusion/tst2.safetensors --vae=/home/user/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors --cache_text_encoder_outputs --xformers --in_json=mdl/metadata_lat.json --lr_scheduler=constant_with_warmup --lr_warmup_steps=100 --save_every_n_epochs=5 --cache_latents --cache_latents_to_disk --noise_offset=0.2 --adaptive_noise_scale=0.02 --learning_rate=4e-7 --train_batch_size=1 --min_snr_gamma=5 --gradient_checkpointing --gradient_accumulation_steps=2 --full_bf16 --optimizer_type=adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
```

## Captions

```
python3 finetune/make_captions.py --batch_size=2 --max_length=100 --min_length=10 ~/sd-scripts/mdl/img

python3 finetune/merge_captions_to_metadata.py --caption_extension=.caption mdl/img mdl/meta_cap.json

python3 finetune/tag_images_by_wd14_tagger.py --batch 4 mdl/img

python3 finetune/merge_dd_tags_to_metadata.py mdl/img --in_json mdl/meta_cap.json mdl/meta_cap_dd.json

python3 finetune/prepare_buckets_latents.py mdl/img mdl/meta_cap_dd.json mdl/metadata_lat.json /home/user/sd-scripts/mdl/insomnia-000008.safetensors --max_resolution=1024,1024 --batch_size=6
```

## Kohya train params for v100
```
accelerate launch  --num_cpu_threads_per_process 8 sdxl_train.py --resolution=1024,1024  --output_dir=mdl --output_name=insomnia --save_precision=fp16 --max_train_steps=5100 --log_with=wandb --log_tracker_config=tracker.toml --logging_dir=mdl/log  --sample_every_n_steps=510 --seed=1234 --sample_prompts=prompt.txt --save_model_as=safetensors --train_data_dir=mdl/img --pretrained_model_name_or_path=/home/user/stable-diffusion-webui/models/Stable-diffusion/tst6.safetensors --vae=/home/user/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors --cache_text_encoder_outputs --xformers --in_json=mdl/metadata_lat.json --lr_scheduler=constant_with_warmup --lr_warmup_steps=10  --save_every_n_epochs=2 --cache_latents --cache_latents_to_disk --full_bf16 --noise_offset=0.1 --adaptive_noise_scale=0.01 --learning_rate=4e-7 --train_batch_size=1 --min_snr_gamma=5 --optimizer_type=adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
```
## Kohya train params for A100 (hi memory, max speed)
```
accelerate launch --num_cpu_threads_per_process 8 sdxl_train.py --resolution=1024,1024  --output_dir=mdl --output_name=insomnia --metadata_title=insomnia --metadata_author=recoilme --metadata_description="telegram: t.me/recoilme" --save_state --save_precision=fp16 --max_train_steps=51000 --log_with=wandb --log_tracker_config=tracker.toml --logging_dir=mdl/log  --sample_every_n_steps=511 --seed=1234 --sample_prompts=prompt.txt --save_model_as=safetensors --train_data_dir=mdl/img --pretrained_model_name_or_path=/home/user/stable-diffusion-webui/models/Stable-diffusion/colorfulxl_v10.safetensors --vae=/home/user/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors --cache_text_encoder_outputs --in_json=mdl/metadata_lat.json --lr_scheduler=constant_with_warmup --lr_warmup_steps=100  --save_every_n_epochs=2 --cache_latents --cache_latents_to_disk --train_text_encoder --noise_offset=0.1 --adaptive_noise_scale=0.01 --learning_rate=4e-7 --train_batch_size=2 --min_snr_gamma=5 --optimizer_type=adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
```

## Kohya train params for A100 (low memory)
```
accelerate launch --num_cpu_threads_per_process 10 sdxl_train.py --mixed_precision=bf16 --full_bf16 --resolution=1024,1024  --output_dir=mdl --output_name=insomnia --metadata_title=insomnia --metadata_author=recoilme --metadata_description="https://t.me/recoilme" --save_state --sdpa --max_token_length=225 --save_precision=fp16 --max_train_steps=51000 --log_with=wandb --log_tracker_config=tracker.toml --logging_dir=mdl/log  --sample_every_n_steps=255 --seed=1234 --sample_prompts=prompt.txt --save_model_as=safetensors --train_data_dir=mdl/img --pretrained_model_name_or_path=/home/user/sd-scripts/mdl/insomnia-000008.safetensors --resume=/home/user/sd-scripts/mdl/insomnia-000008-state --save_last_n_steps_state=2 --vae=/home/user/stable-diffusion-webui/models/VAE/sdxl_vae.safetensors --in_json=mdl/metadata_lat.json --lr_scheduler=constant_with_warmup --lr_warmup_steps=100  --save_every_n_epochs=2 --cache_latents --cache_latents_to_disk --train_text_encoder --noise_offset=0.2 --adaptive_noise_scale=0.02 --learning_rate=4e-7 --train_batch_size=1 --min_snr_gamma=5 --optimizer_type=adafactor --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
```