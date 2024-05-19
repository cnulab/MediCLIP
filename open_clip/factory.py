import json
import logging
import os
import pathlib
import re
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import CLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype
from .openai import load_openai_model
from .transform import image_transform
from .tokenizer import  tokenize


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]

_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def get_tokenizer():
    return tokenize


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys



def create_model(
        model_name: str,
        img_size: int,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
):

    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    pretrained_cfg = {}
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = model_cfg or get_model_config(model_name)

    if model_cfg['vision_cfg']['image_size'] != img_size:
        model_cfg['vision_cfg']['image_size'] = img_size

        cast_dtype = get_cast_dtype(precision)

        model_pre = load_openai_model(
                model_name,
                precision=precision,
                device=device,
        )

        state_dict = model_pre.state_dict()

        model = CLIP(**model_cfg, cast_dtype=cast_dtype)

            ### for resnet
        if not hasattr(model.visual, 'grid_size'):
            model.visual.grid_size = int(np.sqrt(model.visual.attnpool.positional_embedding.shape[0] - 1))

        resize_pos_embed(state_dict, model)
        incompatible_keys = model.load_state_dict(state_dict, strict=True)

        model.to(device=device)

        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    else:
        model = load_openai_model(
                model_name,
                precision=precision,
                device=device,
        )

    model.device=device

    return model,model_cfg



def create_model_and_transforms(
        model_name: str,
        img_size: int,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
):

    model,model_cfg = create_model(
        model_name,
        img_size,
        precision=precision,
        device=device,
    )

    image_mean = OPENAI_DATASET_MEAN
    image_std = OPENAI_DATASET_STD

    preprocess = image_transform(
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess, model_cfg