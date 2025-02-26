import os

import datasets
from datasets import load_dataset


def get_dataset_for_training(tokenizer, path_prefix="./") -> datasets.Dataset:
    def construct_textual_features(example):
        return tokenizer(
            example["data"], example["ref"], truncation=True, max_length=2048
        )

    data_files = {
        "train": os.path.join(path_prefix, "train.csv"),
        "val": os.path.join(path_prefix, "validation.csv"),
    }

    dataset = load_dataset("csv", data_files=data_files, keep_default_na=False)

    tokenized_datasets = dataset.map(
        construct_textual_features,
        batched=True,
        remove_columns=["ref", "data"],
        num_proc=8,
    )
    tokenized_datasets = tokenized_datasets.rename_column("class", "label")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


def get_length_ref(path_prefix="./") -> datasets.Dataset:
    def construct_num_features(example):
        return {"length": len(example["ref"])}

    data_files = {
        "val": os.path.join(path_prefix, "validation.csv"),
    }

    unprocessed_datasets = load_dataset("csv", data_files=data_files)

    unprocessed_datasets = unprocessed_datasets.map(
        construct_num_features, batched=True
    )

    unprocessed_datasets.set_format("torch")
    return unprocessed_datasets
