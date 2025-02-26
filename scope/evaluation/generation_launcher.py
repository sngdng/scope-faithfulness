from copy import deepcopy

import hydra
import torch.distributed as dist
import wandb
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore

from scope.evaluation.evaluation_standard import eval_generation
from scope.scope.alignment.configs import GenConfig
from scope.scope.generate import run_generation_and_evaluation
from scope.utils import read_slurm_env, setup_slurm

cs = ConfigStore.instance()

cs.store(name="base_gen", node=GenConfig)


def get_iterative_parameters(cfg):
    if cfg.model_type == "contrastive":
        it_param = "contrastive_alpha"
    elif cfg.model_type == "mixture":
        it_param = "mixture_alpha"
    elif cfg.model_type == "context-aware":
        it_param = "context_aware_alpha"
    elif cfg.model_type == "critic":
        it_param = "critic_lambda"
    return it_param


@hydra.main(
    version_base=None, config_path="../../configs/generation", config_name="config"
)
def main(cfg: GenConfig):
    load_dotenv()

    setup_slurm()
    rank, local_rank, world_size, devices, num_nodes = read_slurm_env()

    dist.init_process_group(
        backend="mpi", init_method="env://", world_size=world_size, rank=rank
    )
    it_param_name = get_iterative_parameters(cfg)
    it_param = cfg.generation[it_param_name]

    for p in it_param:
        copy_gen_cfg = deepcopy(cfg)
        copy_gen_cfg.generation[it_param_name] = p
        copy_gen_cfg.do_eval = False
        print(f"Running generation for {it_param_name}", p)

        run_generation_and_evaluation(copy_gen_cfg, init_dist=False)
        # Run evaluation only

        if rank == 0:
            hparams = {
                "dataset": copy_gen_cfg.data.path,
                "split": copy_gen_cfg.data.split,
                "model": copy_gen_cfg.model.model_path,
                "noise": copy_gen_cfg.noise.model_path,
                "dtype": copy_gen_cfg.eval.dtype,
            }

            run = wandb.init(
                group=copy_gen_cfg.group,
                name=f'{copy_gen_cfg.model_type}_{copy_gen_cfg.model.model_path}_noise{cfg.noise.model_path}_e{min(it_param)}-{max(it_param)}_{hparams["dataset"]}',
                config=hparams,
                job_type="evaluation",
            )
            wandb.log({it_param_name: p})
            print(f"Running evaluation for {it_param_name}", p)
            eval_generation(copy_gen_cfg, run=run)
        dist.barrier()


if __name__ == "__main__":
    main()
