from src.modules.image_encoder import MSDiffusionImageEncoder
from src.pipeline import MSDiffusionPipeline
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


pretrained_model_name_or_path = "/data00/sqy/checkpoints/stable-diffusion-xl-base-1.0"
clip_model_name_or_path = "/data00/sqy/checkpoints/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
ms_ckpt = "/data00/sqy/checkpoints/MS-Diffusion/ms_adapter.bin"
device = "cuda:2"

lora_weights = "lora-weights/lora-trained-xl-hugging-kp-cross-0.6/pytorch_lora_weights.safetensors"
class_names = ["cat", "plushie bear"]
prompt = f"A cat painted on a stone."
images_path = ["benchmark/cat_5/6.jpg",
               "benchmark/plushie_bear/3.jpg"]
boxes = [[[0.0, 0.203125, 0.546875, 0.859375], [0.546875, 0.265625, 0.9453125, 0.890625]]]
phrases = [class_names]


lora_weights = "lora-weights/lora-trained-xl-fighting-cross-0.6/pytorch_lora_weights.safetensors"
class_names = ["dog", "cat", "plushie bear"]
prompt = f"A cat is fighting with a dog and a plushie bear."
images_path = ["benchmark/dog_0/1.jpg",
               "benchmark/cat_1/1.jpg",
               "benchmark/plushie_bear/3.jpg"]
boxes = [[[0.0, 0.203125, 0.446875, 0.859375],
          [0.446875, 0.214567, 0.696354, 0.87253],
          [0.546875, 0.265625, 0.9453125, 0.890625]]]
phrases = [class_names]


# lora_weights = "lora-weights/lora-trained-xl-riding-cross/pytorch_lora_weights.safetensors"
# class_names = ["elon musk"]
# prompt = f"A elon musk is riding a bike, realism."
# images_path = ["elon_musk.jpg"]
# boxes = [[[0.25, 0.25, 0.75, 0.75]]]
# phrases = [class_names]

# lora_weights = "lora-weights/lora-trained-xl-cooking-cross/pytorch_lora_weights.safetensors"
# class_names = ["person"]
# prompt = f"A {class_names[0]} is cooking."
# images_path = ["elon_musk.jpg"]
# boxes = [[[0.25, 0.25, 0.75, 0.75]]]
# phrases = [class_names]

num_samples = 5
scale = 0.6
load_lora = False

ms_state_dict = torch.load(ms_ckpt)
image_encoder_state_dict = {}
for key, value in ms_state_dict.items():
    if key == "image_proj":
        for k, v in value.items():
            image_encoder_state_dict["resampler." + k] = v
    elif key == "dummy_image_tokens":
        image_encoder_state_dict[key] = value
    else:
        ms_adapter_state_dict = value

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, 
    subfolder="unet"
).to(device, torch.float16)
adapter_modules = set_ms_adapter(unet, scale=scale)
adapter_modules.load_state_dict(ms_adapter_state_dict)

text_encoder_config = CLIPTextConfig.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder"
)

image_encoder = MSDiffusionImageEncoder(
    clip_model_name_or_path,
    dim=1280,
    depth=4,
    dim_head=64,
    heads=20,
    num_queries=16,
    output_dim=unet.config.cross_attention_dim,
    ff_mult=4,
    latent_init_mode="grounding",
    phrase_embeddings_dim=text_encoder_config.projection_dim,
).to(device, dtype=torch.float16)
image_encoder.load_state_dict(image_encoder_state_dict, strict=False)

pipe = MSDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
    image_encoder=image_encoder,
).to(device=device, dtype=torch.float16)

if load_lora:
    pipe.load_lora_weights(lora_weights)

input_images = [Image.open(image_path) for image_path in images_path]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

image_processor = CLIPImageProcessor()
processed_images = [image_processor(images=input_images, return_tensors="pt").pixel_values]
processed_images = torch.stack(processed_images, dim=0) # (1, n, 3, 224, 224)

image_processor_896 = CLIPImageProcessor(
    size=896,
    crop_size=896
)
processed_images_896 = [image_processor_896(images=input_images, return_tensors="pt").pixel_values]
processed_images_896 = torch.stack(processed_images_896, dim=0)  # (1, n, 3, 896, 896)

negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

num = 0
for i in range(32):
    generator = torch.Generator(unet.device).manual_seed(i)
    with torch.no_grad():
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            concept_images=processed_images,
            concept_images_896=processed_images_896,
            num_inference_steps=30,
            boxes=boxes,
            phrases=phrases,
            generator=generator,
            num_images_per_prompt=num_samples
        ).images
        for i in range(len(images)):
            num += 1
            images[i].save(f"test{num}.jpg")