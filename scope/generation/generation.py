import shutil
import warnings

import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GenerationConfig

from scope.utils import (
    SavePathFormat,
    get_dataset,
    get_dtype,
    load_data_collator_model_tokenizer,
    read_slurm_env,
    setup_slurm,
)


def load_data_distributed(config, batch_size, data_collator, rank, world_size):
    dataset = get_dataset(config.data.path)[config.data.split]

    if config.data.get("max_samples", None) is not None:
        warnings.warn(
            "max_samples is set, only using a subset of the data for evaluation. No shuffling, use for debugging only."
        )
        dataset = dataset.select(range(config.data.max_samples))

    chunk_dataset = dataset.shard(num_shards=world_size, index=rank)
    print(f"rank {rank} has {len(chunk_dataset)} samples")

    data_loader = DataLoader(
        dataset=chunk_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=data_collator,
    )

    return data_loader


def prepare_generation_inputs(model, model_type, batch):
    if model_type != "standalone" and model_type != "critic":
        if "layer" in model_type:
            generation_inputs = {
                "input_ids": batch[0]["input_ids"].to(model.device),
                "attention_mask": batch[0]["attention_mask"].to(model.device),
                "weak_inputs": {
                    "input_ids": batch[1]["input_ids"].to(model.device),
                    "attention_mask": batch[1]["attention_mask"].to(model.device),
                },
            }
        else:
            generation_inputs = {
                "input_ids": batch[0]["input_ids"].to(model.device),
                "attention_mask": batch[0]["attention_mask"].to(model.device),
                "weak_inputs": [
                    {
                        "input_ids": weak_batch["input_ids"].to(model.device),
                        "attention_mask": weak_batch["attention_mask"].to(model.device),
                    }
                    for weak_batch in batch[1]
                ],
            }

    else:
        generation_inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
    return generation_inputs


def run_generation(model, config, data_loader):
    generation_config = GenerationConfig(
        return_dict_in_generate=True, output_scores=True, **config.generation
    )
    sequences = []
    dtype = get_dtype(config.eval.dtype)

    sequences_wo_instuctions = []
    print("Running generation")
    it = 0
    for batch in tqdm(data_loader):
        it += 1

        generation_inputs = prepare_generation_inputs(
            model, model_type=config.model_type, batch=batch
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                generation_out = model.generate(
                    generation_config=generation_config,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_inputs,
                )
        if it == 1:
            print(f"batch: {generation_inputs['input_ids'][0]}")
        input_len = generation_inputs["input_ids"].shape[1]

        sequences_wo_instuctions += generation_out.sequences.cpu()[
            :, input_len:
        ].tolist()
        sequences += generation_out.sequences.cpu().tolist()

    # truncate sequences to remove padding from generation
    for i in range(len(sequences_wo_instuctions)):
        if config.generation.eos_token_id in sequences_wo_instuctions[i]:
            stop_idx = sequences_wo_instuctions[i].index(config.generation.eos_token_id)
        else:
            stop_idx = len(sequences_wo_instuctions[i])
        sequences_wo_instuctions[i] = sequences_wo_instuctions[i][: stop_idx + 1]
    return {
        "sequences_instructions": sequences,
        "sequences": sequences_wo_instuctions,
    }


def run_generation_distributed(config, init_dist=True):
    rank, local_rank, world_size, devices, num_nodes = read_slurm_env()

    print(
        f"{rank} process", "ngpus:", torch.cuda.device_count(), "world_size", world_size
    )

    data_collator, model, tokenizer = load_data_collator_model_tokenizer(config)
    if rank == 0:
        save_format = SavePathFormat(config)
        out_path = save_format.get_save_path()
        out_path_object = [out_path]
        tmp_path = save_format.get_tmp_path()
        tmp_path_object = [tmp_path]
    else:
        out_path_object = [None]
        tmp_path_object = [None]
    if init_dist:
        setup_slurm()
        dist.init_process_group(
            backend="mpi", init_method="env://", world_size=world_size, rank=rank
        )
    dist.broadcast_object_list(out_path_object, src=0)
    dist.broadcast_object_list(tmp_path_object, src=0)
    out_path = out_path_object[0]
    tmp_path = tmp_path_object[0]
    print("Saving to:", out_path)

    model.eval()
    data_loader = load_data_distributed(
        config,
        batch_size=config.eval.batch_size,
        data_collator=data_collator,
        rank=rank,
        world_size=world_size,
    )

    generation_dict = run_generation(model, config, data_loader)
    dataset = data_loader.dataset
    dataset = dataset.add_column("generated_ids", generation_dict["sequences"])
    dataset = dataset.add_column(
        "generated_ids_instructions", generation_dict["sequences_instructions"]
    )
    dataset = dataset.map(
        lambda x: {
            "prediction": tokenizer.batch_decode(
                x["generated_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )
    dataset = dataset.map(
        lambda x: {
            "prediction_instructions": tokenizer.batch_decode(
                x["generated_ids_instructions"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )

    print(f"finished rank {rank}")
    if rank == 0:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        dataset.save_to_disk(str(tmp_path / f"predictions_and_scores_{rank}"))
        print("saved rank 0")
    for i in range(1, world_size):
        dist.barrier()
        if rank == i:
            dataset.save_to_disk(str(tmp_path / f"predictions_and_scores_{rank}"))
            print(f"saved rank {rank}")
    dist.barrier()
    if rank == 0:
        list_datasets = []
        for i in range(world_size):
            list_datasets.append(
                load_from_disk(
                    str(tmp_path / f"predictions_and_scores_{i}"),
                )
            )
        dataset_with_results = concatenate_datasets(list_datasets, axis=0)
        dataset_with_results.save_to_disk(str(out_path / "predictions_and_scores"))
        dataset_with_results.to_csv(str(out_path / "predictions_and_scores.csv"))
        print(f"Saved to {out_path / 'predictions_and_scores'}")
        OmegaConf.save(config=config, f=out_path / "config.yaml")
        print_gen_out(dataset_with_results)
        return dataset_with_results


def print_gen_out(dataset):
    for i in range(min(5, len(dataset))):
        sample = dataset[i]

        print("====================================")
        print("Prompt:\n", sample["prompt"])
        print("Real:\n", sample["real"])
        print("------------------------------------")
        print("Gen:\n", sample["prediction"])



