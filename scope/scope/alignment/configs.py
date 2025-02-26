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


from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional

import transformers
from omegaconf import MISSING
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from trl import PPOConfig

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


@dataclass
class ModelConfig:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    tokens_per_word: float = field(
        default=1.31,
        metadata={"help": ("The number of tokens per word to use for the model.")},
    )

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization with PEFT adatpers."
            )
        },
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    ref_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The reference model for generation and loss. None to use model_path."
            )
        },
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The tokenizer base path.")},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    model_code_revision: Optional[str] = field(
        default=None, metadata={"help": "The branch of the IFT model"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    max_length: Optional[int] = 4096
    max_prompt_length: Optional[int] = 256
    context_size: Optional[int] = 4096
    architecture: str = field(
        default="decoder",
        metadata={"help": "Model architecture (decoder or tablellama mainly)."},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "use nested quantization"}
    )
    low_cpu_mem_usage: bool = field(
        default=False, metadata={"help": "use low cpu memory usage"}
    )
    device_map: Optional[str] = field(
        default="auto", metadata={"help": "device mapping"}
    )
    chat_tokens: Optional[List] = field(default=None, metadata={"help": "chat tokens"})
    from_trained: bool = field(
        default=False,
        metadata={"help": "Whether to load the model from a trained model."},
    )
    from_trainer_v2: bool = field(
        default=False,
        metadata={"help": "Whether to load the model from a trainer v2 model."},
    )
    epoch: Optional[int] = field(
        default=None,
        metadata={"help": "The number of epochs the model has been trained."},
    )
    value_head: bool = field(
        default=False,
        metadata={"help": "Whether to use a value head or not."},
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "The attention implementation to use."},
    )
    small_model: bool = field(
        default=False,
        metadata={"help": "Whether to use a small model or not."},
    )
    add_bos: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to add bos token or not.")},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class PEFTConfig:
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    r: Optional[int] = field(
        default=8,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    bias: Optional[str] = field(
        default=None,
        metadata={"help": ("Whether to use bias in the PEFT model.")},
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": ("The task type to use for PEFT.")},
    )


@dataclass
class DataConfig:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        },
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )


@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Used by TRL for reward model training, which tries to read this parameter in init."
            )
        },
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    optim: Optional[str] = field(default="adamw_torch")


@dataclass
class DPOArgument(transformers.TrainingArguments):
    beta: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The beta factor in scope loss. Higher beta means less divergence from the initial policy."
        },
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Used by TRL for reward model training, which tries to read this parameter in init."
            )
        },
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    save_safetensors: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    generation_mode: str = field(
        default="no_context",
        metadata={"help": "The generation mode to use for generation."},
    )
    mixture_alpha: float = field(
        default=0.25,
        metadata={"help": "The mixture alpha to use for generation."},
    )
    mixture_mode: str = field(
        default="hard",
        metadata={"help": "The mixture mode to use for generation."},
    )
    pref_type: str = field(
        default="dpo",
        metadata={"help": "The preference type to use for training."},
    )


@dataclass
class CLIFFArgument(transformers.TrainingArguments):
    tau_value: Optional[float] = field(
        default=1.0,
        metadata={"help": "The tau factor in CLIFF loss."},
    )
    lambda_value: Optional[float] = field(
        default=1.0,
        metadata={"help": "The lambda factor in CLIFF loss."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Used by TRL for reward model training, which tries to read this parameter in init."
            )
        },
    )
    optim: Optional[str] = field(default="rmsprop")
    remove_unused_columns: bool = field(default=False)
    save_safetensors: bool = field(default=False)
    resume_from_checkpoint: Optional[str] = field(default=None)
    generation_mode: str = field(
        default="no_context",
        metadata={"help": "The generation mode to use for generation."},
    )


@dataclass
class GenDataConfig:
    path: str = field(
        default=MISSING,
        metadata={"help": "The path to the dataset to use for generation."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of samples to use for generation."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    split: str = field(
        default=MISSING,
        metadata={"help": "Dataset split"},
    )


@dataclass
class TrainGenConfig:
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "The maximum number of new tokens to generate."},
    )
    min_new_tokens: int = field(
        default=1,
        metadata={"help": "The minimum number of new tokens to generate."},
    )
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to sample or not."},
    )
    pad_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "The pad token id to use for generation."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "The top p value to use for generation."},
    )
    temperature: float = field(
        default=0.6,
        metadata={"help": "The temperature value to use for generation."},
    )
    bos_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "The bos token id to use for generation."},
    )
    eos_token_id: Optional[int] = field(
        default=None,
        metadata={"help": "The eos token id to use for generation."},
    )


@dataclass
class TrainConfig:
    scope: Any = field(default_factory=lambda: {})
    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: TrainGenConfig = field(default_factory=TrainGenConfig)


@dataclass
class GenConfig:
    eval: Any = field(default_factory=lambda: {})
    generation: Any = field(default_factory=lambda: {})
    task: Optional[str] = field(
        default=None,
        metadata={"help": "The task to use for generation."},
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    noise: ModelConfig = field(default_factory=ModelConfig)
    model_type: str = field(
        default="standalone",
        metadata={"help": "The model type to use for generation."},
    )
    data: GenDataConfig = field(default_factory=GenDataConfig)
    with_input: bool = field(
        default=True,
        metadata={"help": "Whether to use input for generation or not."},
    )
    loser_gen: bool = field(
        default=False,
        metadata={"help": "Whether to generate loser samples or not."},
    )
    evaluation_metrics: Optional[Any] = field(
        default=None,
        metadata={"help": "The evaluation metrics to use for generation."},
    )
    offline_setup: bool = field(
        default=False,
        metadata={"help": "Whether to run the evaluation in offline mode or not."},
    )
    group: Optional[str] = field(
        default=None,
        metadata={"help": "The group name for the evaluation run."},
    )
    recompute_eval: bool = field(
        default=False,
        metadata={"help": "Whether to recompute the evaluation or not."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run the evaluation or not."},
    )
    do_gen: bool = field(
        default=True,
        metadata={"help": "Whether to run the generation or not."},
    )
    dist: bool = field(
        default=True,
        metadata={"help": "Whether to run the generation in distributed mode or not."},
    )
    process_context: bool = field(
        default=True,
        metadata={
            "help": "Whether to process the context or not. Set to True for Parent, False otherwise."
        },
    )
