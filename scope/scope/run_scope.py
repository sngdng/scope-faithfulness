#!/usr/bin/env python
#
# Adapted from https://github.com/huggingface/alignment-handbook
import logging
import re
import sys
from pathlib import Path

import hydra
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from alignment import (
    DataConfig,
    DPOArgument,
    DPOTrainer,
    ModelConfig,
    PEFTConfig,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import set_seed

from scope.scope.alignment.configs import TrainConfig
from scope.utils import get_env

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()

cs.store(name="base_train", node=TrainConfig)


def train_scope(cfg):
    model_args = cfg.model
    data_args = cfg.data
    peft_args = cfg.peft
    training_args = cfg.scope
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    RESULT_PATH = get_env("RESULT_PATH")
    hydra_cfg = HydraConfig.get().runtime.choices
    output_dir_name = Path(hydra_cfg.scope) / Path(
        list(cfg.data.dataset_mixer.keys())[0]
    )
    output_dir = RESULT_PATH / output_dir_name
    training_args = DPOArgument(output_dir=output_dir, **training_args)
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"PEFT parameters {peft_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################

    # Truncate from left to ensure we don't lose labels in final turn
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model_path = str(CHECKPOINT_PATH / model_args.model_path)
    if model_args.ref_model_path is None:
        ref_model_path = model_path
    else:
        ref_model_path = str(CHECKPOINT_PATH / model_args.ref_model_path)
    ref_model_kwargs = model_kwargs

    if peft_args.use_peft is True:
        ref_model_path = None
        ref_model_kwargs = None

    #########################
    # Instantiate scope trainer
    #########################

    scope_trainer = DPOTrainer(
        model_path,
        ref_model_path,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=model_args.max_length,  # shouldn't be used if rightly truncated during preprocessing
        max_prompt_length=model_args.max_prompt_length,
        peft_config=get_peft_config(OmegaConf.to_container(peft_args)),
    )

    ###############
    # Training loop
    ###############
    train_result = scope_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    scope_trainer.log_metrics("train", metrics)
    scope_trainer.save_metrics("train", metrics)
    scope_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    scope_trainer.save_model(output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["data-scope"],
        }
        scope_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        scope_trainer.model.config.use_cache = True
        scope_trainer.model.config.save_pretrained(output_dir)

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")
    accelerator.end_training()


@hydra.main(
    version_base=None, config_path="../../configs/training", config_name="config"
)
def main(cfg: TrainConfig):
    train_scope(cfg)


if __name__ == "__main__":
    load_dotenv()
    main()
