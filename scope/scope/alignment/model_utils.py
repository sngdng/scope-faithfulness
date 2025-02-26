# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/alignment-handbook

import os
from typing import Dict

import torch
from accelerate import Accelerator
from huggingface_hub import list_repo_files
from huggingface_hub.utils._validators import HFValidationError
from peft import LoraConfig
from peft import PeftConfig as HFPeftConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer

from scope.utils import get_env

from .configs import DataConfig, ModelConfig, PEFTConfig
from .data import DEFAULT_CHAT_TEMPLATE


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_quantization_config(model_args) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # For consistency with model weights, we use the same value as `torch_dtype` which is float16 for PEFT models
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_tokenizer(
    model_args: ModelConfig, data_args: DataConfig
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH / model_args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    tokenizer.model_max_length = model_args.max_length

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_peft_config(peft_args: PEFTConfig) -> HFPeftConfig | None:
    if peft_args["use_peft"] is False:
        return None

    peft_args.pop("use_peft")
    peft_config = LoraConfig(**peft_args)

    return peft_config


def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except HFValidationError:
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return (
        "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files
    )
