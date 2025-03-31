from .projection import Resampler
from .open_clip.factory import create_model_and_transforms

import os

import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.lora import LoraLoaderMixin


def generate_bboxes(n):
    """
    Generate a list of n*n bounding box coordinates for an image of size 1x1.

    Parameters:
    n (int): Number of divisions along one axis (total bboxes will be n*n).

    Returns:
    bboxes (list): List of bounding box coordinates in the format [x_start, y_start, x_end, y_end].
    """
    bbox_size = 1.0 / n
    bboxes = []
    
    for i in range(n):
        for j in range(n):
            x_start = i * bbox_size
            y_start = j * bbox_size
            x_end = x_start + bbox_size
            y_end = y_start + bbox_size
            bboxes.append([x_start, y_start, x_end, y_end])
    
    return bboxes


class ImageEncoder(ModelMixin, ConfigMixin):
    def __init__(
        self, 
        clip_model_name_or_path,
        dim,
        depth,
        dim_head,
        heads,
        num_queries,
        output_dim,
        ff_mult,
        latent_init_mode,
        phrase_embeddings_dim,
        num_patches=4,
        num_dummy_tokens=4,
        cross_attention_dim=2048,
        clipself_pretrained=None,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            clip_model_name_or_path
        )
        if clipself_pretrained is not None and os.path.exists(clipself_pretrained):
            print("loading local image encoder...")
            model, preprocess_train, preprocess_val = create_model_and_transforms(
                model_name="ViT-bigG-14",
                pretrained=clipself_pretrained,
            )
            self.clipself_model = model
        else:
            self.clipself_model = None
            
        self.resampler = Resampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            embedding_dim=self.clip_model.config.hidden_size,
            output_dim=output_dim,
            ff_mult=ff_mult,
            latent_init_mode=latent_init_mode,
            phrase_embeddings_dim=phrase_embeddings_dim
        )
        self.dummy_image_tokens = nn.Parameter(torch.randn(1, num_dummy_tokens, cross_attention_dim))
    
    def forward(self, concept_images, concept_images_896, grounding_kwargs):
        concept_images = concept_images
        bsz, n, c, h, w = concept_images.shape
        concept_images = concept_images.reshape(-1, c, h, w)
        concept_embeddings = self.clip_model(concept_images, output_hidden_states=True).hidden_states[-2]
        
        if self.clipself_model is not None:
            concept_images_896 = concept_images_896.reshape(-1, c, 896, 896)
            normed_boxes = [torch.tensor(generate_bboxes(self.num_patches),
                                         dtype=concept_images.dtype, 
                                         device=concept_images.device).repeat(bsz * n, 1)]
            concept_embeddings_4096 = self.clipself_model.encode_pseudo_boxes(
                concept_images_896,
                normed_boxes=normed_boxes,
                extract_type="v2")
            concept_embeddings = torch.cat([concept_embeddings, concept_embeddings_4096.reshape(bsz * n, -1, 1664)], dim=1)
            
        image_prompt_embeds = self.resampler(concept_embeddings, grounding_kwargs)
        image_prompt_embeds = image_prompt_embeds.view(bsz, -1, image_prompt_embeds.shape[-2], image_prompt_embeds.shape[-1])  # (bsz, rn, num_tokens, cross_attention_dim)
        image_prompt_embeds = image_prompt_embeds.view(bsz, image_prompt_embeds.shape[-3] * image_prompt_embeds.shape[-2],
                                                        image_prompt_embeds.shape[-1])  # (bsz, total_num_tokens * rn, cross_attention_dim)
        dummy_image_tokens = self.dummy_image_tokens.repeat(bsz, 1, 1)
        image_prompt_embeds = torch.cat([dummy_image_tokens, image_prompt_embeds], dim=1)
         
        return image_prompt_embeds