# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from pathlib import Path

import hydra
from datasets import DatasetDict
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore

from scope.evaluation.evaluation_standard import eval_generation
from scope.generation.generation import run_generation_distributed
from scope.scope.alignment.configs import GenConfig
from scope.utils import (
    SavePathFormat,
    get_dataset,
    get_env,
    read_slurm_env,
)


def post_process(x):
    return x


cs = ConfigStore.instance()

cs.store(name="base_gen", node=GenConfig)


def get_output_dir(cfg):
    save_path_format = SavePathFormat(cfg)
    model_folder_name = save_path_format.get_model_folder_name()
    data = cfg.data.path
    output_dir = Path(data)
    output_dir = output_dir / model_folder_name
    return output_dir


def run_generation_and_evaluation(cfg: GenConfig, init_dist=True):
    load_dotenv()
    if cfg.do_gen:
        print(f"Launching generation.")
        rank, local_rank, world_size, devices, num_nodes = read_slurm_env()
        if cfg.loser_gen:
            cfg.data.split = "train"

        gen_dataset = run_generation_distributed(cfg, init_dist=init_dist)

        if cfg.loser_gen:
            output_dir = get_output_dir(cfg)
            RESULT_PATH = get_env("DATA_PATH")
            output_dir = RESULT_PATH / output_dir

            if rank == 0:
                output_dir.mkdir(parents=True, exist_ok=True)

                gen_dataset = gen_dataset.select_columns(
                    ["prompt_no_input", "prompt", "prediction", "real"]
                )
                gen_dataset = gen_dataset.rename_column("prediction", "generated")

                train_dataset = gen_dataset.map(post_process)
                test_dataset = get_dataset(cfg.data.path)["test"]
                val_dataset = get_dataset(cfg.data.path)["validation"]
                out_dataset = DatasetDict(
                    {
                        "train": train_dataset,
                        "test": test_dataset,
                        "validation": val_dataset,
                    }
                )
                out_dataset.save_to_disk(str(output_dir))
                print(f"Saved to {output_dir}")
    if cfg.do_eval and not cfg.loser_gen:
        rank, local_rank, world_size, devices, num_nodes = read_slurm_env()
        if rank == 0:
            eval_generation(cfg)


@hydra.main(
    version_base=None, config_path="../../configs/generation", config_name="config"
)
def main(cfg: GenConfig):
    run_generation_and_evaluation(cfg)


if __name__ == "__main__":
    main()
