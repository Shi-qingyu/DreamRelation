import re
import torch
from diffusers import UNet2DConditionModel

from src.modules.attention_processor import MaskedIPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor


def get_phrase_idx(tokenizer, phrase, prompt, get_last_word=False, num=0):
    def is_equal_words(pr_words, ph_words):
        if len(pr_words) != len(ph_words):
            return False
        for pr_word, ph_word in zip(pr_words, ph_words):
            if "-"+ph_word not in pr_word and ph_word != re.sub(r'[.!?,:]$', '', pr_word):
                return False
        return True

    phrase_words = phrase.split()
    if len(phrase_words) == 0:
        return [0, 0], None
    if get_last_word:
        phrase_words = phrase_words[-1:]
    # prompt_words = re.findall(r'\b[\w\'-]+\b', prompt)
    prompt_words = prompt.split()
    start = 1
    end = 0
    res_words = phrase_words
    for i in range(len(prompt_words)):
        if is_equal_words(prompt_words[i:i+len(phrase_words)], phrase_words):
            if num != 0:
                # skip this one
                num -= 1
                continue
            end = start
            res_words = prompt_words[i:i+len(phrase_words)]
            res_words = [re.sub(r'[.!?,:]$', '', w) for w in res_words]
            prompt_words[i+len(phrase_words)-1] = res_words[-1]  # remove the last punctuation
            for j in range(i, i+len(phrase_words)):
                end += len(tokenizer.encode(prompt_words[j])) - 2
            break
        else:
            start += len(tokenizer.encode(prompt_words[i])) - 2

    if end == 0:
        return [0, 0], None

    return [start, end], res_words


def get_eot_idx(tokenizer, prompt):
    words = prompt.split()
    start = 1
    for w in words:
        start += len(tokenizer.encode(w)) - 2
    return start

def set_ms_adapter(unet: UNet2DConditionModel, scale=0.6, weight_dtype=torch.float16, num_tokens=16, text_tokens=77):
    # set attention processor
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=scale,
                num_tokens=num_tokens,
                text_tokens=text_tokens,
            ).to(unet.device, dtype=weight_dtype)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules
