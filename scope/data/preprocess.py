import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from datasets import DatasetDict, load_dataset
from scope.tasks import (
    CNNDMTask,
    DARTTask,
    E2ETask,
    FeTaQATask,
    HighlightedFeTaQATask,
    TableInstructTask,
    TottoTask,
    WebNLGTask,
    XSumTask,
)
from scope.utils import get_env


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="preprocessed")
    parser.add_argument("--data", type=str, default="GEM/totto")
    parser.add_argument("--chat_tokens", nargs="*", type=str, default=None)
    return parser.parse_args()


def load_and_process_data_ultrachat(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    preprocessed_data = [
        {
            "generated": [
                message["messages"][0],
                {"role": "assistant", "content": ""},
            ],
            "real": [message["messages"][0], message["messages"][1]],
        }
        for message in dataset
    ]
    return preprocessed_data


def load_and_process_data_tableInstruct(dataset_name, split):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    tableInstruct_task = TableInstructTask()
    preprocessed_data = []
    for sample in tqdm(dataset):
        preprocessed_data.append(
            {
                "prompt_no_input": tableInstruct_task.format_input(
                    sample, with_input=False
                ),
                "prompt": tableInstruct_task.format_input(sample, with_input=True),
                "generated": "",
                "real": sample["output"],
            }
        )
    return preprocessed_data


def load_and_process_data_totto(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    totto_task = TottoTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": totto_task.format_input(sample, with_input=False),
            "prompt": totto_task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["target"],
            "linearized_input": sample["linearized_input"],
        }
        if split == "validation" or split == "test":
            source, parent_table = totto_task.format_eval(sample)
            sample_dict["source"] = source
            sample_dict["parent_table"] = parent_table
            if split == "validation":
                sample_dict["references"] = sample["references"]
        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_fetaqa(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split)
    task = FeTaQATask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["answer"],
        }

        if "test" in split or "validation" in split:
            source, parent_table = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["references"] = [sample["answer"]]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_highlightedfetaqa(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split)
    task = HighlightedFeTaQATask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["answer"],
        }

        if "test" in split or "validation" in split:
            source, parent_table = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["references"] = [sample["answer"]]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_webnlg(dataset_name, subset, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    task = WebNLGTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["target"],
        }
        if "test" in split or "validation" in split:
            source, parent_table = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["references"] = sample["references"]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_dart(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    task = DARTTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["target"],
        }
        if "test" in split or "validation" in split:
            source, parent_table, entity = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["entity"] = entity
            sample_dict["references"] = sample["references"]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_webnlg(dataset_name, subset, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    task = DARTTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["target"],
        }
        if "test" in split or "validation" in split:
            source, parent_table = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["references"] = sample["references"]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_e2e(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    task = E2ETask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["target"],
        }
        if "test" in split or "validation" in split:
            source, parent_table = task.format_eval(sample)
            sample_dict["parent_table"] = parent_table
            sample_dict["source"] = source
            sample_dict["references"] = sample["references"]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_xsum(dataset_name, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    task = XSumTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["summary"],
        }
        if "test" in split or "validation" in split:
            sample_dict["source"] = sample["document"]
            sample_dict["references"] = [sample["summary"]]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def load_and_process_data_cnndm(dataset_name, subset, split, chat_tokens=None):
    dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
    task = CNNDMTask(chat_tokens=chat_tokens)
    preprocessed_data = []
    for sample in tqdm(dataset):
        sample_dict = {
            "prompt_no_input": task.format_input(sample, with_input=False),
            "prompt": task.format_input(sample, with_input=True),
            "generated": "",
            "real": sample["highlights"],
        }
        if "test" in split or "validation" in split:
            sample_dict["source"] = sample["article"]
            sample_dict["references"] = [sample["highlights"]]

        preprocessed_data.append(sample_dict)

    return preprocessed_data


def save_to_json(data, path):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")


def main():
    load_dotenv()
    args = setup_arg_parser()
    DATA_PATH = get_env("DATA_PATH")
    output_dir = DATA_PATH / args.output_dir
    if args.chat_tokens is not None:
        output_dir = Path(f"{output_dir}_chat")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing {args.data} dataset...")
    elif args.data == "GEM/totto":
        train_data = load_and_process_data_totto(
            args.data, "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_totto(
            args.data, "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_totto(
            args.data, "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "fetaqa":
        train_data = load_and_process_data_fetaqa(
            "DongfuJiang/FeTaQA", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_fetaqa(
            "DongfuJiang/FeTaQA", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_fetaqa(
            "DongfuJiang/FeTaQA", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "highlightedfetaqa":
        train_data = load_and_process_data_highlightedfetaqa(
            "DongfuJiang/FeTaQA", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_highlightedfetaqa(
            "DongfuJiang/FeTaQA", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_highlightedfetaqa(
            "DongfuJiang/FeTaQA", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "webnlg":
        train_data = load_and_process_data_webnlg(
            "GEM/web_nlg", "en", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_webnlg(
            "GEM/web_nlg", "en", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_webnlg(
            "GEM/web_nlg", "en", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "dart":
        train_data = load_and_process_data_dart(
            "GEM/dart", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_dart(
            "GEM/dart", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_dart(
            "GEM/dart", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "e2e":
        train_data = load_and_process_data_e2e(
            "GEM/e2e_nlg", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_e2e(
            "GEM/e2e_nlg", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_e2e(
            "GEM/e2e_nlg", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "xsum":
        train_data = load_and_process_data_xsum(
            "EdinburghNLP/xsum", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_xsum(
            "EdinburghNLP/xsum", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_xsum(
            "EdinburghNLP/xsum", "test", chat_tokens=args.chat_tokens
        )
    elif args.data == "cnndm":
        train_data = load_and_process_data_cnndm(
            "abisee/cnn_dailymail", "3.0.0", "train", chat_tokens=args.chat_tokens
        )
        val_data = load_and_process_data_cnndm(
            "abisee/cnn_dailymail", "3.0.0", "validation", chat_tokens=args.chat_tokens
        )
        test_data = load_and_process_data_cnndm(
            "abisee/cnn_dailymail", "3.0.0", "test", chat_tokens=args.chat_tokens
        )
    else:
        logging.error(f"Invalid dataset: {args.data}")
        return

    train_json_path = output_dir / "train.json"
    test_json_path = output_dir / "test.json"

    save_to_json(train_data, train_json_path)
    save_to_json(test_data, test_json_path)

    dataset = load_dataset("json", data_files=str(train_json_path), split="train")
    dataset_test = load_dataset("json", data_files=str(test_json_path), split="train")

    if val_data:
        val_json_path = output_dir / "val.json"
        save_to_json(val_data, val_json_path)
        dataset_val = load_dataset("json", data_files=str(val_json_path), split="train")
        dataset_dict = DatasetDict(
            {"train": dataset, "validation": dataset_val, "test": dataset_test}
        )
    else:
        dataset_dict = DatasetDict({"train": dataset, "test": dataset_test})
    dataset_dict.save_to_disk(str(output_dir))
    # print some samples
    for i in range(5):
        print("=" * 20)
        print(dataset_dict["train"][i]["prompt"])
        print("-" * 20)
        print(dataset_dict["train"][i]["prompt_no_input"])
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
