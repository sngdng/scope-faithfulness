import argparse
import os
from pathlib import Path

import spacy_stanza
import torch
from dotenv import load_dotenv
from multiprocess import set_start_method
from transformers import BartForConditionalGeneration, BartTokenizer

from datasets import load_from_disk


def mask_entities_and_relations(nlp, tokenizer, text):
    # Process the input text to get the dependency parse
    doc = nlp(text)
    masked_text = text
    # Iterate over each token in the processed document
    for token in doc:
        head_token = token.head
        # Check if the token has a dependency relation of 'obl',
        # is an entity, and the head token is a NOUN, VERB, ADJ, or ADV
        if (
            token.dep_ == "obl"
            and token.ent_type_  # Check if the token is part of an entity
            and head_token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        ):
            # Mask the token in the text
            masked_text = (
                masked_text[: token.idx]
                + tokenizer.mask_token
                + masked_text[token.idx + len(token.text) :]
            )
            # Mask the head token in the text
            masked_text = (
                masked_text[: head_token.idx]
                + tokenizer.mask_token
                + masked_text[head_token.idx + len(head_token.text) :]
            )
    # Mask all the entities in the text
    for ent in doc.ents:
        masked_text = masked_text.replace(ent.text, tokenizer.mask_token)
    return masked_text


def main():
    load_dotenv()
    DATA_PATH = Path(os.getenv("DATA_PATH"))
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="path to pos bt dataset")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-c", "--column", type=str, default="real")
    args = parser.parse_args()

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    dataset = load_from_disk(DATA_PATH / f"{args.dataset}")
    train_dataset = dataset["train"]
    nlp = spacy_stanza.load_pipeline("en", download_method=None)

    def mask_text(batch):
        masked_texts = [
            mask_entities_and_relations(nlp, tokenizer, text)
            for text in batch[args.column]
        ]
        return {"masked_text": masked_texts}

    train_dataset = train_dataset.map(
        mask_text,
        batched=True,
        batch_size=args.batch_size,
    )
    models = []
    for rank in range(torch.cuda.device_count()):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"

        model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large"
        ).eval()
        new_model = model.to(device)
        models.append(new_model)

    def process_batch(batch, rank):
        inputs = tokenizer(
            batch["masked_text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(models[rank].device)
        outputs = models[rank].generate(
            **inputs,
            max_length=512,
            min_length=10,
            early_stopping=False,
            do_sample=True,
            top_k=50,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"neg": generated_texts}

    set_start_method("spawn")
    train_dataset = train_dataset.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=torch.cuda.device_count(),
    )
    dataset["train"] = train_dataset
    dataset.save_to_disk(DATA_PATH / f"{args.dataset}_neg_mask")


if __name__ == "__main__":
    main()
