import math
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import hostlist
import torch
from torch import nn
from torch.optim import Optimizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from datasets import load_from_disk
from scope.critic.lmodule import ClassificationModule
from scope.critic.model import SimpleClassifierModelWithBNSELU
from scope.decoding import (
    ContrastiveDecoder,
    ContrastiveDecoderConfig,
    CriticDecoder,
    CriticDecoderConfig,
    MixtureDecoder,
    PMIDecoder,
    PMIDecoderConfig,
)
from scope.preprocessing import ExpertGuidedDataCollator, LLama2PromptDataCollator


def load_expert_guided_decoder(config, main_model, load_model=True):
    if config.model_type == "critic":
        CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
        model = SimpleClassifierModelWithBNSELU(
            CHECKPOINT_PATH / config.generation.critic_base_model_path
        )
        state_dict = torch.load(
            CHECKPOINT_PATH / config.generation.critic_model_path,
        )["state_dict"]
        critic_model = ClassificationModule(model)
        critic_model.load_state_dict(state_dict, strict=False)

        critic_model.eval()
        critic_tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT_PATH / config.generation.critic_base_model_path
        )
        main_tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT_PATH / config.model.model_path
        )
        expert_guided_config = CriticDecoderConfig(
            vocab_size=main_model.config.vocab_size,
            decoder_start_token_id=main_model.config.decoder_start_token_id,
            lambd=config.generation.critic_lambda,
            critic_top_k=config.generation.critic_top_k,
            linear_warmup=config.generation.critic_linear_warmup,
        )
        critic_model.to(main_model.device)

        model = CriticDecoder(
            main_model,
            main_tokenizer,
            critic_model,
            critic_tokenizer,
            config=expert_guided_config,
            max_length=config.model.max_length,
        )
        return model

    main_config = config.model
    noise_config = config.noise

    assert (
        noise_config is not None
    ), "You need to supply either a noise model or a weak model."

    if load_model:
        if noise_config.model_path == main_config.model_path:
            print(f"Using same backbone for noise model")
            noise_model = main_model
        else:
            CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
            model_path = CHECKPOINT_PATH / config.noise.model_path
            dtype = get_dtype(config.eval.dtype)
            attn_implementation = noise_config.attn_implementation
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
            noise_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                pad_token_id=tokenizer.pad_token_id,
                low_cpu_mem_usage=True,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
            )

    else:
        noise_model = None

    if not load_model:
        return None
    else:
        model_type = config.model_type
        if model_type == "mixture":
            model = MixtureDecoder(
                model=main_model,
                unconditional_model=noise_model,
                mixture_alpha=config.generation.mixture_alpha,
                mixture_mode=config.generation.mixture_mode,
            )
            return model
        elif model_type == "context-aware":
            model = MixtureDecoder(
                model=main_model,
                unconditional_model=noise_model,
                mixture_alpha=config.generation.context_aware_alpha,
                mixture_mode="cad",
            )
            return model

        else:
            if model_type == "contrastive":
                expert_guided_config = ContrastiveDecoderConfig(
                    vocab_size=main_model.config.vocab_size,
                    decoder_start_token_id=main_model.config.decoder_start_token_id,
                    alpha=config.generation.contrastive_alpha,
                )
                model_class = ContrastiveDecoder
            elif model_type == "pmi":
                expert_guided_config = PMIDecoderConfig(
                    vocab_size=main_model.config.vocab_size,
                    decoder_start_token_id=main_model.config.decoder_start_token_id,
                    lambd=config.generation.pmi_lambda,
                    tau=config.generation.pmi_tau,
                )
                model_class = PMIDecoder

            else:
                raise ValueError(f"Model type {model_type} not recognized")

            expert_guided_decoder = model_class(
                main_model, noise_model, config=expert_guided_config
            )

            return expert_guided_decoder


def load_task_config(config, with_input=True):
    task_name = config.task
    if task_name == "totto":
        from scope.tasks import TottoTask

        task = TottoTask(with_input=with_input, chat_tokens=config.model.chat_tokens)
    elif "-gen" in task_name:
        from scope.tasks import GenerationTask

        task = GenerationTask(with_input=with_input)

    else:
        raise ValueError(f"Task {task_name} not recognized")
    return task


def load_data_collator_model_tokenizer(config, load_model=True):
    model, tokenizer = get_model_and_tokenizer(config, load_model=load_model)
    task = load_task_config(config, with_input=config.with_input)

    data_collator = LLama2PromptDataCollator(
        tokenizer=tokenizer,
        task_formater=task,
        max_length=config.model.max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    if config.model_type != "standalone":
        assert (
            config.noise is not None
        ), "Noise model must be provided in guided decoding system."
        model = load_expert_guided_decoder(config, model, load_model=load_model)
        # hardcoding with_input=False for our use case
        if config.model_type != "critic":
            noise_task = load_task_config(config, with_input=False)
            noise_data_collator = LLama2PromptDataCollator(
                tokenizer=tokenizer,
                task_formater=noise_task,
                max_length=config.noise.max_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )

            data_collator = ExpertGuidedDataCollator(data_collator, noise_data_collator)
    model.eval()
    return data_collator, model, tokenizer


def get_dtype(type_str):
    if type_str == "bf16":
        return torch.bfloat16
    elif type_str == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_env(env_variable):
    return Path(os.environ[env_variable])


def get_dataset(data_path):
    DATA_PATH = get_env("DATA_PATH")
    return load_from_disk(str(DATA_PATH / data_path))


def read_slurm_env():
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    devices = int(os.environ.get("SLURM_GPUS_ON_NODE", "0"))
    num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    return rank, local_rank, world_size, devices, num_nodes


def setup_slurm():
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ.get("SLURM_JOB_NODELIST", ""))

    # get IDs of reserved GPU
    gpu_ids = os.environ.get("SLURM_STEP_GPUS", None)
    if gpu_ids is not None:
        gpu_ids = gpu_ids.split(",")
        port_complement = int(min(gpu_ids))
    else:
        port_complement = 0

    # define MASTER_ADD & MASTER_PORT
    if len(hostnames) > 1:
        os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ["MASTER_PORT"] = str(12345 + port_complement)
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12345 + port_complement)


class AdamWScale(Optimizer):
    """
    This AdamW implementation is copied from Huggingface.
    We modified it with Adagrad scaling by rms of a weight tensor

    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                beta1 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # /Adapt Step from Adafactor
                step_size = step_size * max(1e-3, self._rms(p.data))
                # /Adapt Step from Adafactor

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


def get_model_and_tokenizer(cfg, load_model=True):
    CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
    if cfg.model.from_trained:
        model_folder = get_env("RESULT_PATH")
        model_path = model_folder / cfg.model.model_path

    else:
        model_path = CHECKPOINT_PATH / cfg.model.model_path
    tokenizer_path = model_path

    dtype = get_dtype(cfg.eval.dtype)

    attn_implementation = cfg.model.get("attn_implementation", "flash_attention_2")

    if cfg.model.architecture == "decoder":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if not cfg.model.add_bos:
            tokenizer.bos_token = None
            tokenizer.bos_token_id = None

        if load_model:
            if cfg.model.value_head:
                model_class = AutoModelForCausalLMWithValueHead
            else:
                model_class = AutoModelForCausalLM
            if cfg.model.small_model:
                model_kwargs = {"device_map": "cuda"}
            else:
                model_kwargs = {
                    "device_map": "auto",
                    "attn_implementation": attn_implementation,
                }

            model = model_class.from_pretrained(
                str(model_path),
                pad_token_id=tokenizer.pad_token_id,
                torch_dtype=dtype,
                **model_kwargs,
            )

            # model = torch.compile(model, mode="reduce-overhead")
        else:
            model = None
    elif cfg.model.architecture == "tablellama":
        # Set RoPE scaling factor
        config = AutoConfig.from_pretrained(
            model_path,
        )

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        model_ctx_len = getattr(cfg.model, "context_size", None)
        if orig_ctx_len and model_ctx_len and model_ctx_len > orig_ctx_len:
            scaling_factor = float(math.ceil(model_ctx_len / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # Load model and tokenizer
        dtype = get_dtype(cfg.eval.dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            model_max_length=(
                model_ctx_len if model_ctx_len > orig_ctx_len else orig_ctx_len
            ),
            padding_side="left",
            truncation_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=attn_implementation,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        raise ValueError(f"Model architecture {cfg.model.architecture} not recognized")
    return model, tokenizer


class SavePathFormat:
    """
    Format the path for saving and loading the results.
    """

    def __init__(self, config):
        self.config = config

    def get_tmp_path(self):
        TMP_PATH = get_env("TMPDIR")
        print(f"TMPDIR: {TMP_PATH}")
        job_id = os.environ.get("SLURM_JOB_ID", "0")
        current_date = datetime.now().strftime("%m-%d-%H:%M:%S")
        task_name = self.get_task_name()
        return TMP_PATH / task_name / f"{job_id}_{current_date}"

    @staticmethod
    def get_model_name(model_config):
        model_path = model_config.model_path
        epoch = model_config.epoch
        if epoch is not None:
            model_path = str(Path(model_path).parent / f"epoch-{epoch}")
        return model_path.replace("_", "-")

    def get_task_name(self):
        return self.config.task

    def get_model_folder_name(self):
        folder_name = (
            f"{self.config.model_type}/{self.get_model_name(self.config.model)}/"
        )
        if self.config.model_type == "mixture":
            folder_name = (
                folder_name
                + f"{self.config.generation.mixture_mode}_a{self.config.generation.mixture_alpha}"
            )
        if self.config.model_type == "context-aware":
            folder_name = folder_name + f"{self.config.generation.context_aware_alpha}"
        if self.config.model_type == "contrastive":
            folder_name = folder_name + f"{self.config.generation.contrastive_alpha}"
        if self.config.model_type == "pmi":
            folder_name = (
                folder_name
                + f"_l{self.config.generation.pmi_lambda}_t{self.config.generation.pmi_tau}"
            )
        if self.config.model_type == "critic":
            folder_name = folder_name + f"_l{self.config.generation.critic_lambda}"

        if (
            self.config.model_type != "standalone"
            and self.config.model_type != "critic"
        ):
            noise_model_name = self.get_model_name(self.config.noise)
            folder_name = folder_name + "_" + f"noise{noise_model_name}"

        return folder_name

    def get_path(self):
        RESULT_PATH = get_env("RESULT_PATH")
        split_path = self.config.data.path.split("/")
        if len(split_path) > 1:
            out_path = RESULT_PATH / split_path[-2] / split_path[-1]
        else:
            out_path = RESULT_PATH / split_path[-1]

        folder_name = self.get_model_folder_name()
        task_name = self.get_task_name()
        folder_name = Path(folder_name) / task_name

        if self.config.data.split == "test":
            out_path = out_path / "test"

        return out_path / folder_name

    def get_save_path(self):
        out_path = self.get_path()
        current_time = datetime.now().strftime("%m-%d-%H:%M:%S")
        out_path = out_path / current_time
        out_path.mkdir(exist_ok=True, parents=True)
        return out_path

    def get_results_path(self, date="latest"):
        out_path = self.get_path()

        if date == "latest":
            # Get latest timestamp in folder
            folders = [f for f in os.listdir(out_path)]
            timestamps_datetime = [
                datetime.strptime(ts, "%m-%d-%H:%M:%S") for ts in folders
            ]
            latest_folder = max(timestamps_datetime).strftime("%m-%d-%H:%M:%S")
        else:
            latest_folder = date

        out_path = out_path / latest_folder
        return out_path

    def get_generation_results_path(self, date="latest"):
        return self.get_results_path(date) / "predictions_and_scores"
