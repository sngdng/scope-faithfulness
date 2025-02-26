import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import spacy_stanza
import torch
from dotenv import load_dotenv
from multiprocess import set_start_method
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from datasets import load_from_disk


def swap_entities(nlp, text, entities):
    # Process the input text to get the dependency parse
    doc = nlp(text)
    swapped_text = text
    # Iterate over each entity in the processed document and replace it with a random entity in entities
    for ent in doc.ents:
        swapped_text = swapped_text.replace(
            ent.text,
            entities[ent.label_][random.randint(0, len(entities[ent.label_]) - 1)],
        )
    return swapped_text


def main():
    load_dotenv()
    random.seed(42)
    DATA_PATH = Path(os.getenv("DATA_PATH"))
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to dataset")
    parser.add_argument("-c", "--column", type=str, default="real")
    args = parser.parse_args()

    dataset = load_from_disk(DATA_PATH / f"{args.dataset}")
    train_dataset = dataset["train"]
    nlp = spacy_stanza.load_pipeline("en", download_method=None)
    all_entities = defaultdict(list)
    for doc in tqdm(nlp.pipe(train_dataset[args.column], batch_size=256)):
        for ent in doc.ents:
            all_entities[ent.label_].append(ent.text)
    train_dataset = train_dataset.map(
        lambda x: {"generated": swap_entities(nlp, x[args.column], all_entities)}
    )

    dataset["train"] = train_dataset
    dataset.save_to_disk(DATA_PATH / f"{args.dataset}_swap_ent")


if __name__ == "__main__":
    main()
