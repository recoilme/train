import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import subprocess
import sys

import gc
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from contextlib import nullcontext
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

import lora_sdxl_TI

from lora_sdxl import *

logger = get_logger(__name__)



def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
        
        
def merge_lora_models(models, merge_dtype, model_path):
    
    def load_state_dict(file_name, dtype):
        if os.path.splitext(file_name)[1] == ".safetensors":
            sd = load_file(file_name)
        else:
            sd = torch.load(file_name, map_location="cuda")
        for key in list(sd.keys()):
            if type(sd[key]) == torch.Tensor:
                sd[key] = sd[key].to(dtype)
        return sd
    
    def save_to_file(file_name, model, dtype):
        if dtype is not None:
            for key in list(model.keys()):
                if type(model[key]) == torch.Tensor:
                    model[key] = model[key].to(dtype)

        if os.path.splitext(file_name)[1] == ".safetensors":
            save_file(model, file_name)
        else:
            torch.save(model, file_name)


    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    for model in models:
        #print(f"loading: {model}")
        lora_sd = load_state_dict(model, torch.float32)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        #print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        #print(f"merging...")
        for key in lora_sd.keys():
            if "alpha" in key:
                continue

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha)

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size()
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    #print("merged model")
    #print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    save_to_file(model_path, merged_sd, torch.float16)


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
    else:
        sd = torch.load(file_name, map_location="cuda")
    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)
    return sd


def save_to_file(file_name, model, dtype):
    if dtype is not None:
        for key in list(model.keys()):
            if type(model[key]) == torch.Tensor:
                model[key] = model[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name)
    else:
        torch.save(model, file_name)
        
        
                
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="~/dream/stable-diffusion-XL",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="~/dream/img",
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/dream/out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    parser.add_argument(
        "--save_n_steps",
        type=int,
        default=1,
        help=("Save the model every n global_steps"),
    )
    
    
    parser.add_argument(
        "--save_starting_step",
        type=int,
        default=1,
        help=("The step from which it starts saving intermediary checkpoints"),
    )
    
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=1000000,
        help=("The step at which the text_encoder is no longer trained"),
    )


    parser.add_argument(
        "--image_captions_filename",
        action="store_true",
        help="Get captions from filename",
    )    
    
      
    
    parser.add_argument(
        "--Resumetr",
        type=str,
        default="False",        
        help="Resume training info",
    )    
    
 
    
    parser.add_argument(
        "--Session_dir",
        type=str,
        default="~/dream/ses",     
        help="Current session directory",
    )    

    parser.add_argument(
        "--external_captions",
        action="store_true",
        default=False,        
        help="Use captions stored in a txt file",
    )    
    
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="",
        help="The folder where captions files are stored",
    )        

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help="Offset Noise",
    )
    
    parser.add_argument(
        "--ofstnselvl",
        type=float,
        default=0.03,        
        help="Offset Noise amount",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training",
    )    
    
    parser.add_argument(
        "--dim",
        type=int,
        default=16,        
        help="LoRa dimension",
    )    

    args = parser.parse_args()
    
    return args



class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        args,
        tokenizers,        
        text_encoders,
        size=512,
        center_crop=False,
        instance_prompt_hidden_states=None,
        instance_unet_added_conditions=None,    
    ):
        self.size = size
        self.tokenizers=tokenizers
        self.text_encoders=text_encoders
        self.center_crop = center_crop
        self.instance_prompt_hidden_states = instance_prompt_hidden_states
        self.instance_unet_added_conditions = instance_unet_added_conditions
        self.image_captions_filename = None

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if args.image_captions_filename:
            self.image_captions_filename = True
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index, args=parse_args()):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        if self.image_captions_filename:
            filename = Path(path).stem
            
            pt=''.join([i for i in filename if not i.isdigit()])
            pt=pt.replace("_"," ")
            pt=pt.replace("(","")
            pt=pt.replace(")","")
            pt=pt.replace("-","")
            pt=pt.replace("conceptimagedb","")  
            
            if args.external_captions:
              cptpth=os.path.join(args.captions_dir, filename+'.txt')
              if os.path.exists(cptpth):
                with open(cptpth, "r") as f:
                    instance_prompt=f.read()
              else:
                instance_prompt=pt
            else:
                instance_prompt = pt
        
        example["instance_images"] = self.image_transforms(instance_image)
        with torch.no_grad():
            example["instance_prompt_ids"], example["instance_added_cond_kwargs"]= compute_embeddings(args, instance_prompt, self.text_encoders, self.tokenizers)
        return example

            
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )
            
        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds    
    

def collate_fn(examples):

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]        
    add_text_embeds = [example["instance_added_cond_kwargs"]["text_embeds"] for example in examples]
    add_time_ids = [example["instance_added_cond_kwargs"]["time_ids"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).half()

    input_ids = torch.cat(input_ids, dim=0)
    add_text_embeds = torch.cat(add_text_embeds, dim=0)
    add_time_ids = torch.cat(add_time_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }

    return batch


def compute_embeddings(args, prompt, text_encoders, tokenizers):
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    crops_coords_top_left = (0, 0)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to('cuda')
        add_text_embeds = add_text_embeds.to('cuda')
        add_time_ids = add_time_ids.to('cuda', dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return prompt_embeds, unet_added_cond_kwargs
    
    
class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache, cond_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.cond_cache = cond_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index], self.cond_cache[index]
    
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
          
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )
        
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
        use_auth_token=True,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
        use_auth_token=True
    )

     
    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    
    # Load scheduler and models

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=True,
    ).to("cuda")
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", use_auth_token=True
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=True).to('cuda')
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=True
    ).to('cuda')
    
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.eval()
    text_encoder_two.eval()
    vae.eval()
    unet.eval()
    
    model_path = os.path.join(args.Session_dir, os.path.basename(args.Session_dir) + ".safetensors")
    model_path_TI = os.path.join(args.Session_dir, os.path.basename(args.Session_dir) + "_TI.safetensors")
    network = create_network(1, args.dim, 20000, unet)
    if args.resume:
        network.load_weights(model_path)

    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    set_diffusers_xformers_flag(unet, True)   
   
    network.apply_to(unet, True)
    network.prepare_optimizer_params(args.learning_rate)
    trainable_params = network.parameters()
   
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    
    if os.path.exists(model_path_TI):    
        for weights_file in [model_path_TI]:
            if ";" in weights_file:
                weights_file, multiplier = weights_file.split(";")
                multiplier = float(multiplier)
            else:
                multiplier = 1.0

            lora_model, weights_sd = lora_sdxl_TI.create_network_from_weights(
                multiplier, weights_file, text_encoders, None, True
            )
            lora_model.merge_to(text_encoders, weights_sd, torch.float32, 'cuda')    
    
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_auth_token=True)
    
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
        size=args.resolution,
        center_crop=args.center_crop,
        args=args
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        network, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    network.prepare_grad_etc(network)    

    
    latents_cache = []
    text_encoder_cache = []
    cond_cache= []
    for batch in train_dataloader:
        with torch.no_grad():
            
            batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
            batch["unet_added_conditions"] = batch["unet_added_conditions"]

            batch["pixel_values"]=(vae.encode(batch["pixel_values"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor)

            latents_cache.append(batch["pixel_values"])
            text_encoder_cache.append(batch["input_ids"])
            cond_cache.append(batch["unet_added_conditions"])

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache, cond_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    del vae, tokenizers, text_encoders
    gc.collect()
    torch.cuda.empty_cache()   
    
        
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def bar(prg):
       br='|'+'█' * prg + ' ' * (25-prg)+'|'
       return br
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            network.train()
            with accelerator.accumulate(network):
                with torch.no_grad():
                    model_input = batch[0][0]
 
                # Sample noise that we'll add to the latents
                if args.offset_noise:  
                    noise = torch.randn_like(model_input)

                else:
                    noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()

                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Predict the noise residual
                with accelerator.autocast():
                    model_pred = unet(noisy_model_input, timesteps, batch[0][1], added_cond_kwargs=batch[0][2]).sample

                # Get the target for loss depending on the prediction type
                target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
            fll=round((global_step*100)/args.max_train_steps)
            fll=round(fll/4)
            pr=bar(fll)
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            progress_bar.set_description_str("Progress")
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
         network = accelerator.unwrap_model(network)
    accelerator.end_training()
    
    
    if os.path.exists(model_path_TI):
        network.save_weights(model_path, torch.float16, None)
        final_models=[model_path, model_path_TI]
        merge_lora_models(final_models, torch.float16, model_path)
        subprocess.call('rm '+model_path_TI, shell=True)
    else:
        network.save_weights(model_path, torch.float16, None)
    
      
    accelerator.end_training()

if __name__ == "__main__":
    main()
