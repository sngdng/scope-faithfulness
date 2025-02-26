import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize as split_text_into_sentences
from tqdm import tqdm

load_dotenv()
data_path = sys.argv[1]
text_col = sys.argv[2]
input_col = sys.argv[3]
split_name = sys.argv[4]
DATA_PATH = Path(os.environ["DATA_PATH"])
OUTPUT = DATA_PATH / "critic" / data_path / f"{split_name}.csv"


dataset = load_from_disk(str(DATA_PATH / data_path))[split_name]
dataset = dataset.shuffle(seed=42)


def get_random_sentence(triplets, text_col):
    while True:
        line = random.choice(dataset)
        if line[input_col] == triplets:
            continue
        sentences = split_text_into_sentences(line[text_col])
        return random.choice(sentences)


def get_other_related_sentence(triplets, ref, text_col, input_col):
    other_refs = [
        line
        for line in dataset
        if line[input_col] == triplets and line[text_col] != ref
    ]
    if len(other_refs) == 0:
        return None
    new_ref = random.choice(other_refs)
    sentences = split_text_into_sentences(new_ref[text_col])
    return random.choice(sentences)


def has_many_trip(line, input_col):
    return line[input_col].find(";") != -1


def how_many_sent(line, text_col):
    sentences = split_text_into_sentences(line[text_col])
    return len(sentences)


def write_neg_line(line, txt, text_col, input_col, max_rows_per_line=10):
    rows = []
    txt_corr = line[text_col].strip().split(" ")
    txt = txt.strip().split(" ")
    for i in range(len(txt)):
        if txt[: (i + 1)] != txt_corr[: (i + 1)]:
            row = [" ".join(txt[: (i + 1)])]
            row.append(line[input_col])
            row.append(0)
            rows.append(row)
    random.shuffle(rows)
    rows = rows[:max_rows_per_line]

    return rows


def write_line(line, clazz, text_col, input_col, txt=None):
    rows = []
    if txt is None:
        txt = line[text_col].strip().split(" ")
    else:
        txt = txt.strip().split(" ")
    for i in range(len(txt)):
        row = [" ".join(txt[: (i + 1)])]
        row.append(line[input_col])
        row.append(clazz)
        rows.append(row)
    return rows


def process_line(line, text_col, input_col):
    results = []

    # correct examples
    results.extend(write_line(line, 1, text_col, input_col))

    # continue generation
    sent = get_other_related_sentence(
        line[input_col], line[text_col], text_col, input_col
    )
    if sent is None:
        sent = get_random_sentence(line[input_col], text_col)
    if sent is not None:
        results.extend(
            write_neg_line(line, line[text_col] + " " + sent, text_col, input_col)
        )

    # replace sentence with another
    sent = get_random_sentence(line[input_col], text_col)
    if sent is not None:
        sentences = split_text_into_sentences(line[text_col])
        if sentences:  # Ensure sentences is not empty
            random_idx = random.randint(0, len(sentences) - 1)
            sentences[random_idx] = sent
            results.extend(
                write_neg_line(line, " ".join(sentences), text_col, input_col)
            )

    # replace a random word
    if sent is not None:
        words = line[text_col].split(" ")
        if len(words) > 1:  # Ensure words has more than one word
            random_word = random.choice(sent.split(" "))
            random_idx = random.randint(1, len(words) - 1)
            words[random_idx] = random_word
            results.extend(write_neg_line(line, " ".join(words), text_col, input_col))

    return results


all_rows = []

with ProcessPoolExecutor() as executor:
    for result in tqdm(
        executor.map(
            partial(process_line, text_col=text_col, input_col=input_col), dataset
        ),
        total=len(dataset),
    ):
        all_rows.extend(result)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="UTF-8") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ref", "data", "class"])
    writer.writerows(all_rows)
