import argparse
import os
from omegaconf import OmegaConf
from src.modules.image_encoder import ImageEncoder
from src.pipeline import Pipeline
from src.utils import set_ms_adapter
import torch
from PIL import Image
from safetensors import safe_open
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPTextConfig

def load_lora_weights(ckpt_path):
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    return state_dict

def main():
    parser = argparse.ArgumentParser(description="Inference Script")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    
    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Pretrained model name or path")
    parser.add_argument("--clip_model_name_or_path", type=str,
                        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                        help="CLIP model name or path")
    parser.add_argument("--ms_ckpt", type=str,
                        default="./checkpoints/MS-Diffusion/ms_adapter.bin",
                        help="Path to MS adapter checkpoint")
    parser.add_argument("--clipself_ckpt", type=str,
                        default="./checkpoints/local_image_encoder/epoch_6.pt",
                        help="Path to CLIP self checkpoint")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--scale", type=float, default=0.6,
                        help="Scale parameter for MS adapter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps")
    
    # Boolean flags
    parser.add_argument("--load_lora", action="store_true",
                        help="Whether to load LoRA weights")
    parser.add_argument("--no_load_lora", dest="load_lora", action="store_false",
                        help="Disable loading LoRA weights")
    parser.set_defaults(load_lora=True)
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for generated images")
    parser.add_argument("--negative_prompt", type=str,
                        default="monochrome, lowres, bad anatomy, worst quality, low quality",
                        help="Negative prompt for generation")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run the model on")
    
    args = parser.parse_args()
    
    # Load config from the provided path
    config = OmegaConf.load(args.config)
    
    # Load MS adapter state dict
    ms_state_dict = torch.load(args.ms_ckpt)
    image_encoder_state_dict = {}
    for key, value in ms_state_dict.items():
        if key == "image_proj":
            for k, v in value.items():
                image_encoder_state_dict["resampler." + k] = v
        elif key == "dummy_image_tokens":
            image_encoder_state_dict[key] = value
        else:
            ms_adapter_state_dict = value
    
    # Initialize UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet"
    ).to(args.device, torch.float16)
    adapter_modules = set_ms_adapter(unet, scale=args.scale)
    adapter_modules.load_state_dict(ms_adapter_state_dict)
    
    # Initialize text encoder config
    text_encoder_config = CLIPTextConfig.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    
    # Initialize image encoder
    image_encoder = ImageEncoder(
        args.clip_model_name_or_path,
        clipself_pretrained=args.clipself_ckpt,
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=16,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        latent_init_mode="grounding",
        phrase_embeddings_dim=text_encoder_config.projection_dim,
    ).to(args.device, dtype=torch.float16)
    image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
    
    # Initialize pipeline
    pipe = Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
    ).to(device=args.device, dtype=torch.float16)
    
    # Load LoRA weights if specified
    if args.load_lora:
        pipe.load_lora_weights(config.lora_weights)
    
    # Process input images
    input_images = [Image.open(image_path) for image_path in config.images_path]
    input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]
    
    image_processor = CLIPImageProcessor()
    processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
    processed_images = torch.stack(processed_images, dim=0)  # (1, n, 3, 224, 224)
    
    image_processor_896 = CLIPImageProcessor(
        size=896,
        crop_size=896
    )
    processed_images_896 = [image_processor_896(images=input_images, return_tensors="pt").pixel_values]
    processed_images_896 = torch.stack(processed_images_896, dim=0)  # (1, n, 3, 896, 896)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate image
    generator = torch.Generator(unet.device).manual_seed(args.seed)
    with torch.no_grad():
        image = pipe(
            prompt=config.prompt,
            negative_prompt=args.negative_prompt,
            concept_images=processed_images,
            concept_images_896=processed_images_896,
            num_inference_steps=args.num_inference_steps,
            boxes=config.boxes,
            phrases=config.phrases,
            generator=generator,
            num_images_per_prompt=args.num_samples
        ).images[0]
        image.save(os.path.join(args.output_dir, "output.jpg"))

if __name__ == "__main__":
    main()