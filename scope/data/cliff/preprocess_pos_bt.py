import argparse
import os
import re

import countryinfo
import nlpaug.augmenter.word as naw
import spacy
import torch
from dotenv import load_dotenv
from multiprocess import set_start_method

from datasets import load_from_disk

nlp = spacy.load("en_core_web_sm")


QUOTE_RE = re.compile(r'".+?"')
OTHER_QUOTE_RE = re.compile(r"'.+?'")


demonym_map = {}
with open("scope/data/cliff/demonyms.txt") as f:
    for line in f:
        country, d1, d2 = line.strip().split("\t")
        d1 = d1.split(", ")
        d2 = d2.split(", ")
        val = d1 + d2 + [country]
        demonym_map[country] = val
        for x in d1 + d2:
            demonym_map[x] = val


def check_word(ent, other_accepted, ori):
    for tok in ent:
        if tok.pos_ in ["DET", "ADP", "PUNCT"]:
            continue

        if tok.text == "'s":
            continue

        tok_text = tok.text

        if tok_text.lower() in ori.lower():
            continue

        if tok_text in other_accepted:
            continue

        alt_text = tok_text[:-1]
        if tok_text.endswith("s") and alt_text.lower() in ori.lower():
            continue

        try:
            cinfo = countryinfo.CountryInfo(tok.text).info()
            if "name" in cinfo and (
                cinfo["name"] in ori or cinfo["name"] in other_accepted
            ):
                continue
            if "altSpellings" in cinfo and (
                any([spell in ori for spell in cinfo["altSpelling"]])
                or any([spell in other_accepted for spell in cinfo["altSpelling"]])
            ):
                continue
            if "demonym" in cinfo and (
                cinfo["demonym"] in ori or cinfo["demonym"] in other_accepted
            ):
                continue
        except KeyError:
            pass

        if any(
            [
                demo in ori or demo in other_accepted
                for demo in demonym_map.get(tok.text, [])
            ]
        ):
            continue

        return False

    return True


def check_one(aug, ori):
    tmp_ori = ori.replace("'s ", " ")
    ori_quotes = QUOTE_RE.findall(tmp_ori)
    ori_quotes = [quote[1:-1].strip() for quote in ori_quotes]

    tmp_aug = aug.replace("'s ", " ")
    aug_quotes = QUOTE_RE.findall(tmp_aug) + OTHER_QUOTE_RE.findall(tmp_aug)
    aug_quotes = [quote[1:-1].strip() for quote in aug_quotes]

    if ori_quotes and aug_quotes:
        if any([quote not in ori_quotes for quote in aug_quotes]):
            return "quote"

    aug_doc = nlp(aug)
    ori_doc = nlp(ori)

    other_accept_by_ori = set()
    for ent in ori_doc.ents:
        if ent.label_ in [
            "CARDINAL",
            "DATE",
            "MONEY",
            "ORDINAL",
            "PERCENT",
            "QUANTITY",
            "TIME",
        ]:
            continue
        ent_text = ent.text
        if ent_text.endswith("'s"):
            ent_text = ent_text[:-2]
        elif ent_text.endswith("s'"):
            ent_text = ent_text[:-1]
        try:
            cinfo = countryinfo.CountryInfo(ent_text).info()
            if "name" in cinfo:
                cname = cinfo["name"]
                other_accept_by_ori.add(cname)
            if "altSpellings" in cinfo:
                other_accept_by_ori.update(cinfo["altSpellings"])
            if "demonym" in cinfo:
                other_accept_by_ori.add(cinfo["denonym"])
        except KeyError:
            pass
        other_accept_by_ori.update(demonym_map.get(ent_text, []))

        if ent_text.lower().startswith("the"):
            alt_text = ent_text[4:]
            try:
                cinfo = countryinfo.CountryInfo(alt_text).info()
                if "name" in cinfo:
                    cname = cinfo["name"]
                    other_accept_by_ori.add(cname)
                if "altSpellings" in cinfo:
                    other_accept_by_ori.update(cinfo["altSpellings"])
                if "demonym" in cinfo:
                    other_accept_by_ori.add(cinfo["denonym"])
            except KeyError:
                pass
            other_accept_by_ori.update(demonym_map.get(alt_text, []))

        for tok in ent:
            if tok.pos_ in ["DET", "ADP", "PUNCT"]:
                continue
            try:
                cinfo = countryinfo.CountryInfo(tok.text).info()
                if "name" in cinfo:
                    cname = cinfo["name"]
                    other_accept_by_ori.add(cname)
                if "altSpellings" in cinfo:
                    other_accept_by_ori.update(cinfo["altSpellings"])
                if "demonym" in cinfo:
                    other_accept_by_ori.add(cinfo["denonym"])
            except KeyError:
                pass

            other_accept_by_ori.update(demonym_map.get(tok.text, []))

    other_accept_by_ori = " ".join(list(other_accept_by_ori))

    for ent in aug_doc.ents:
        if ent.label_ in [
            "CARDINAL",
            "DATE",
            "MONEY",
            "ORDINAL",
            "PERCENT",
            "QUANTITY",
            "TIME",
        ]:
            continue
        ent_text = ent.text
        if ent_text.endswith("'s"):
            ent_text = ent_text[:-2]
        elif ent_text.endswith("s'"):
            ent_text = ent_text[:-1]

        if all(
            [
                tok.text.lower() in ori.lower()
                for tok in ent
                if tok.pos_ not in ["DET", "ADP", "PUNCT"] and tok.text != "'s"
            ]
        ):
            continue

        if ent_text.lower() in ori.lower():
            continue

        if ent_text in other_accept_by_ori:
            continue

        try:
            cinfo = countryinfo.CountryInfo(ent_text).info()
            if "name" in cinfo and (
                cinfo["name"] in ori or cinfo["name"] in other_accept_by_ori
            ):
                continue
            if "altSpellings" in cinfo and (
                any([spell in ori for spell in cinfo["altSpelling"]])
                or any([spell in other_accept_by_ori for spell in cinfo["altSpelling"]])
            ):
                continue
            if "demonym" in cinfo and (
                cinfo["demonym"] in ori or cinfo["demonym"] in other_accept_by_ori
            ):
                continue
        except KeyError:
            pass

        if any(
            [
                demo in ori or demo in other_accept_by_ori
                for demo in demonym_map.get(ent_text, [])
            ]
        ):
            continue

        if ent_text.lower().startswith("the"):
            alt_text = ent_text[4:]
            try:
                cinfo = countryinfo.CountryInfo(alt_text).info()
                if "name" in cinfo and (
                    cinfo["name"] in ori or cinfo["name"] in other_accept_by_ori
                ):
                    continue
                if "altSpellings" in cinfo and (
                    any([spell in ori for spell in cinfo["altSpelling"]])
                    or any(
                        [spell in other_accept_by_ori for spell in cinfo["altSpelling"]]
                    )
                ):
                    continue
                if "demonym" in cinfo and (
                    cinfo["demonym"] in ori or cinfo["demonym"] in other_accept_by_ori
                ):
                    continue
            except KeyError:
                pass

            if any(
                [
                    demo in ori or demo in other_accept_by_ori
                    for demo in demonym_map.get(alt_text, [])
                ]
            ):
                continue

        if check_word(ent, other_accept_by_ori, ori):
            continue

        # print(ent_text)
        return f"entity {ent_text}"

    return "correct"


def main():
    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset name")
    parser.add_argument("-n", "--num_samples", type=int, default=-1)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-c", "--column", type=str, default="real")
    args = parser.parse_args()

    print("Loading dataset")
    dataset = load_from_disk(f"{DATA_PATH}/{args.dataset}")
    train_dataset = dataset["train"]
    if args.num_samples > 0:
        train_dataset = train_dataset.select(range(args.num_samples))
    models = []
    # moving model to device
    for rank in range(torch.cuda.device_count()):
        print("Loading back translation model on rank", rank)
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        back_translation_aug = naw.BackTranslationAug(
            from_model_name="facebook/wmt19-en-de",
            to_model_name="facebook/wmt19-de-en",
            batch_size=args.batch_size,
            device=device,
        )
        models.append(back_translation_aug)

    def augment_and_filter(batch, rank):
        bt = models[rank].augment(batch[args.column])

        return {"bt": bt}

    set_start_method("spawn")
    train_dataset = train_dataset.map(
        augment_and_filter,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True,
        num_proc=torch.cuda.device_count(),  # one process per GPU
    )
    dataset["train"] = train_dataset
    dataset.save_to_disk(f"{DATA_PATH}/{args.dataset}_bt")
    print("Done")


if __name__ == "__main__":
    main()
