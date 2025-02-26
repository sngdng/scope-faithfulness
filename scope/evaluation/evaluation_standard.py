from argparse import ArgumentParser

import hydra
import pandas as pd
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from scope.evaluation.scores import DatasetScorer
from scope.scope.alignment.configs import GenConfig
from scope.utils import SavePathFormat


def add_context_to_dataset(x):
    source = x["source"]
    try:
        name = source[0].split(" | ")[0]
        context = name
        for triple in source:
            sub, prp, obj = triple.split(" | ")
            context += f"\n\n- {prp}: {obj}"
    except:
        context = source
    return {"context": context}


def add_first_reference_to_dataset(x):
    first_reference = x["references"][0] if "references" in x else ""
    return {"first_reference": first_reference}


def get_wandb_hparams_to_log(config):
    model_name = SavePathFormat(config).get_model_folder_name()
    hparams = {
        "dataset": config.data.path,
        "split": config.data.split,
        "model": model_name,
        "dtype": config.eval.dtype,
    }
    return hparams


def load_generation_results(dataset_path, split_by=None, process_context=True):
    print("Loading from:", dataset_path)
    dataset = load_from_disk(str(dataset_path))

    # process triples (sanity check)
    def process_triples(sample):
        new_triples = []
        for t in sample["source"]:
            s = t.split(" | ")
            if len(s) > 3:
                # remove | after the first 3 elements
                new_triple = " | ".join(s[:3]) + " ".join(s[3:])
                new_triples.append(new_triple)
            else:
                # leave triple unchanged
                new_triples.append(t)
        sample["source"] = new_triples
        return sample

    if process_context:
        dataset = dataset.map(process_triples)
    if split_by is not None:
        dataset = dataset.map(
            lambda x: {"prediction": x["prediction"].split(split_by)[0]}
        )
    return dataset


def eval_generation(cfg, run=None):
    dataset_path = SavePathFormat(cfg).get_generation_results_path()
    if "pythia" in str(dataset_path):
        split_by = "\n"
    else:
        split_by = None

    dataset = load_generation_results(
        dataset_path, process_context=cfg.process_context, split_by=split_by
    )

    scorer = DatasetScorer(
        file=dataset,
        scorers_names=cfg.evaluation_metrics,
        offline_setup=cfg.offline_setup,
    )
    if cfg.offline_setup:
        print("Offline setup done")
    else:
        if run is None:
            hparams = get_wandb_hparams_to_log(cfg)
            name = f'{hparams["model"]}_{hparams["dataset"]}'
            if "test" in cfg.data.split:
                name = f"[TEST]{name}"
            run = wandb.init(
                group=cfg.group,
                name=name,
                config=hparams,
                job_type="evaluation",
                mode="offline",
            )
        save_path = (
            dataset_path.parent
            / f"RES_{dataset_path.stem}_{'-'.join(cfg.evaluation_metrics)}"
        )
        if save_path.exists() and not cfg.recompute_eval:
            print("Loading from:", save_path)
            evaluation_results = load_from_disk(str(save_path))
        else:
            evaluation_results = scorer.score_dataset()
            print("Saving to:", save_path)
            evaluation_results.save_to_disk(str(save_path))

        evaluation_results = evaluation_results.map(
            add_context_to_dataset, load_from_cache_file=False
        )
        evaluation_results = evaluation_results.map(
            add_first_reference_to_dataset, load_from_cache_file=False
        )
        writable_results = scorer.get_writable_columns(evaluation_results.to_dict())

        df_results = pd.DataFrame(writable_results)
        df_results.to_csv(f"{save_path}.csv")

        wandb_table = wandb.Table(dataframe=df_results)
        wandb.log({"samples": wandb_table})

        # Log the config, a bit hacky but works
        wandb_config_table = wandb.Table(
            columns=["config"], data=[[OmegaConf.to_yaml(cfg)]]
        )
        wandb.log({"config": wandb_config_table})

        average_results = scorer.get_average_results(evaluation_results.to_dict())
        print("Average results:")
        df = pd.DataFrame.from_dict({k: [v] for k, v in average_results.items()})
        df = df.reindex(sorted(df.columns), axis=1)
        df.index = ["score"]
        print(df.round(4).T)
        wandb.log(average_results)


cs = ConfigStore.instance()

cs.store(name="base_gen", node=GenConfig)


@hydra.main(
    version_base=None, config_path="../../configs/generation", config_name="config"
)
def main(cfg: GenConfig):
    load_dotenv()
    eval_generation(cfg)


if __name__ == "__main__":
    main()
