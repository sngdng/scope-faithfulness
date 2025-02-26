import logging
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import evaluate
import numpy as np
import openai
import spacy
import torch
import torch.nn as nn
from bert_score import BERTScorer
from datasets import load_dataset
from nltk import FreqDist, ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from parent import parent
from questeval.questeval_metric import QuestEval
from sentence_transformers import CrossEncoder
from transformers import BartForConditionalGeneration, BartTokenizer

from scope.utils import get_env

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def word_tokenize(*args, **kwargs):
    list_words = nltk_word_tokenize(*args, **kwargs)
    if list_words == []:
        list_words = [""]
    return list_words


def get_spacy_pipeline():
    try:
        spacy_pipeline = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("Downloading language model for the spaCy model.")
        from spacy.cli import download

        download("en_core_web_sm")
        spacy_pipeline = spacy.load("en_core_web_sm")
    return spacy_pipeline


class Triple:
    def __init__(
        self,
        raw_text: str,
        spacy_pipeline=None,
        lower: bool = False,
    ):
        sbj, prp, obj = self.safe_split(raw_text)
        if spacy_pipeline is not None:
            obj = " ".join(
                [t.text for t in spacy_pipeline(self.clean_obj(obj.strip(), lc=lower))]
            )
            prp = self.clean_prp(prp.strip())
            sbj = " ".join(
                [t.text for t in spacy_pipeline(self.clean_obj(sbj.strip(), lc=lower))]
            )
            if prp == "ethnicgroup":
                obj = obj.split("_in_")[0]
                obj = obj.split("_of_")[0]
        else:
            obj = obj.strip()
            prp = prp.strip()
            sbj = sbj.strip()

        self.sbj = sbj
        self.obj = obj
        self.prp = prp

    @staticmethod
    def safe_split(raw_text) -> List[str]:
        if not isinstance(raw_text, str):
            raise TypeError(
                'A triple must be a string with two "|"' f"but you gave: {raw_text}"
            )

        split = raw_text.strip().split(" | ")
        if not len(split) == 3:
            raise TypeError(
                'A triple must be a string with two "|"' f"but you gave: {raw_text}"
            )

        return split

    def __repr__(self):
        return f"{self.sbj} | {self.prp} | {self.obj}"

    @staticmethod
    def clean_obj(s, lc: bool = False):
        s = unidecode.unidecode(s)
        if lc:
            s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub("_", " ", s)  # turn undescores to spaces
        return s

    @staticmethod
    def clean_prp(s: str, lc: bool = False) -> str:
        s = unidecode.unidecode(s)
        if lc:
            s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub("\s+", "_", s)  # turn spaces to underscores
        s = re.sub("\s+\(in metres\)", "_m", s)
        s = re.sub("\s+\(in feet\)", "_f", s)
        s = re.sub("\(.*\)", "", s)
        s = re.sub(" \| ", "\|", s)  # remove vert bar for later templating
        return s.strip()


class BaseScorer:
    fields: List[str]

    def offline_setup(self):
        return

    def score(
        self,
        dataset,
        **kwargs,
    ):
        return

    def return_average(self, dict_results):
        return {k: np.mean(dict_results[k]) for k in self.fields}


class RougeScorer(BaseScorer):
    fields = ["rouge.rouge1", "rouge.rouge2", "rouge.rougeLsum", "rouge.meanrouge"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scorer = evaluate.load(
            str(get_env("METRICS_PATH") / "rouge"), keep_in_memory=True
        )

    def score(self, dataset):
        results = self.scorer.compute(
            predictions=dataset["prediction"],
            references=dataset["references"],
            use_aggregator=False,
        )
        # compute mean rouge
        results["meanrouge"] = []
        for i in range(len(results["rouge1"])):
            results["meanrouge"].append(
                (results["rouge1"][i] + results["rouge2"][i] + results["rougeLsum"][i])
                / 3.0
            )

        results = {f"rouge.{k}": v for k, v in results.items()}

        return {k: results[k] for k in self.fields}


class DataQuestEvalScorer(BaseScorer):
    fields = ["dataquest"]

    def __init__(self, task: str = "data2text", **kwargs):
        super().__init__(**kwargs)
        self.task = task

    def offline_setup(self):
        QuestEval(task=self.task, list_scores=("answerability",))

    def score(self, dataset):
        questeval = QuestEval(
            task=self.task, list_scores=("answerability",), use_cache=False
        )
        score = questeval.corpus_questeval(
            sources=dataset["source"],
            hypothesis=dataset["prediction"],
            batch_size=256,
        )

        results = {"dataquest": score["ex_level_scores"]}
        return results


class BertScorer(BaseScorer):
    fields = ["bert.p", "bert.r", "bert.f1"]

    def __init__(self, bert_model_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if bert_model_path is None:
            self.bert_model_path = "roberta-large"
        else:
            self.bert_model_path = bert_model_path

    def offline_setup(self):
        BERTScorer(
            model_type=self.bert_model_path,
            lang="en",
            rescale_with_baseline=True,
            nthreads=1,
        )

    def score(self, dataset):
        scorer = BERTScorer(
            model_type=self.bert_model_path,
            lang="en",
            rescale_with_baseline=True,
            nthreads=1,
        )
        P, R, F1 = scorer.score(cands=dataset["prediction"], refs=dataset["references"])
        results = {
            "bert.f1": F1.cpu(),
            "bert.p": P.cpu(),
            "bert.r": R.cpu(),
        }

        results = {k: v.tolist() for k, v in results.items()}

        return results


class ParentScorer(BaseScorer):
    fields = ["parent.r"]

    def __init__(
        self,
        lower: bool = False,
        data_format: str = "table",
    ) -> None:
        super().__init__()
        self.spacy_pipeline = get_spacy_pipeline()
        self.lower = lower
        self.data_format = data_format

    def format_parent(self, dataset):
        table = []
        for x in dataset["source"]:
            res = []
            for t in x:
                triple = Triple(
                    t,
                    spacy_pipeline=self.spacy_pipeline,
                    lower=self.lower,
                )
                if self.data_format == "table":
                    data_tuple = (
                        [triple.prp],
                        word_tokenize(triple.obj),
                    )
                elif self.data_format == "graph":
                    data_tuple = (
                        [triple.prp],
                        word_tokenize(triple.sbj) + word_tokenize(triple.obj),
                    )
                else:
                    raise ValueError(f"Unknown data format: {self.data_format}")
                res.append(data_tuple)
            table.append(res)
        return table

    def score(self, dataset):
        parent_predictions = [word_tokenize(pred) for pred in dataset["prediction"]]
        if "references" in dataset.column_names:
            parent_references = [
                [word_tokenize(ref) for ref in subref]
                for subref in dataset["references"]
            ]
        else:
            parent_references = [[""]] * len(dataset)
        if "parent_table" not in dataset.column_names:
            parent_sources = self.format_parent(dataset)
        else:
            parent_sources = [
                [([prp], word_tokenize(obj)) for (prp, obj) in table]
                for table in dataset["parent_table"]
            ]
        _, parent_recall, _ = parent(
            parent_predictions,
            parent_references,
            parent_sources,
            avg_results=False,
            use_tqdm=True,
            n_jobs=-1,
            lambda_weight=1.0,  # to remove ref from the recall
        )

        return {"parent.r": parent_recall}


class SacreBLEUScorer(BaseScorer):
    fields = ["sacrebleu"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sacrebleu_metric = evaluate.load(
            str(get_env("METRICS_PATH") / "sacrebleu"), keep_in_memory=True
        )

    def score(self, dataset):
        results = {}
        references = dataset["references"]
        # Compute SacreBLEU
        # (since HF SacreBLEU only support references with same length,
        # we have to pad them into the same length)
        ref_max_len = max([len(ref) for ref in references])
        # see https://github.com/mjpost/sacrebleu/pull/132
        sacrebleu_references = [
            references + [None] * (ref_max_len - len(references))
            for references in references
        ]

        results[f"sacrebleu"] = (
            self.sacrebleu_metric.compute(
                predictions=dataset["prediction"], references=sacrebleu_references
            )["score"]
            / 100.0
        )
        results = {k: [v] * len(dataset) for k, v in results.items()}
        return results


class AlignScorer(BaseScorer):
    fields = ["align"]

    def __init__(self, **kwargs):
        from alignscore import AlignScore

        CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
        self.scorer = AlignScore(
            model="roberta-base",
            batch_size=32,
            device="cuda:0",
            ckpt_path=str(CHECKPOINT_PATH / "align-score.ckpt"),
            evaluation_mode="nli_sp",
        )

    def score(self, dataset):
        results = self.scorer.score(
            contexts=dataset["source"], claims=dataset["prediction"]
        )
        return {"align": results}


class BARTScorer(BaseScorer):
    fields = ["bart"]

    def __init__(
        self, device="cuda:0", max_length=1024, checkpoint="facebook/bart-large-cnn"
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )
        self.lsm = nn.LogSoftmax(dim=1)

    def score(self, dataset):
        """Score a batch of examples"""
        batch_size = 8
        srcs = dataset["source"]
        tgts = dataset["prediction"]
        score_list = []

        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i : i + batch_size]
            tgt_list = tgts[i : i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    )
                    src_tokens = encoded_src["input_ids"].to(self.device)
                    src_mask = encoded_src["attention_mask"].to(self.device)

                    tgt_tokens = encoded_tgt["input_ids"].to(self.device)
                    tgt_mask = encoded_tgt["attention_mask"]
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens, attention_mask=src_mask, labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f"source: {src_list}")
                print(f"target: {tgt_list}")
                exit(0)
        results = {"bart": score_list}
        return results


class MNLIScorer(BaseScorer):
    """
    Compute the MNLI score adapted for Data2Text from (Dusek et al. 2020).
    We consider the minimum entailment score for both omissions and
    hallucinations as the MNLI score for a prediction.
    """

    fields = [
        "mnli.omission_probs",
        "mnli.hallucination_probs",
        "mnli.ok",
        "mnli.omission",
        "mnli.hallucination",
        "mnli.omission_hallucination",
        "mnli.score",
    ]

    def __init__(self, mnli_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if mnli_path is None:
            self.mnli_path = "cross-encoder/nli-deberta-v3-large"
        else:
            self.mnli_path = mnli_path

    def offline_setup(self):
        model = CrossEncoder(self.mnli_path)

    @staticmethod
    def softmax(x):
        """
        Compute the softmax function for a given array of real numbers.

        Parameters:
        x (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Softmax probabilities.
        """
        # Subtract the maximum value for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

        # Compute softmax probabilities
        softmax_probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return softmax_probs

    def score(self, dataset):
        results = defaultdict(list)
        CHECKPOINT_PATH = get_env("CHECKPOINT_PATH")
        model = CrossEncoder(CHECKPOINT_PATH / self.mnli_path)
        for triples, pred in zip(dataset["source"], dataset["prediction"]):
            premise_hypothesises = [
                (pred, f"Text is about {triples[0].split(' | ')[0]}.")
            ]
            hypothesises = [f"Text is about {triples[0].split(' | ')[0]}. "]
            for triple in triples:
                key, value = (
                    triple.split(" | ")[1].strip(),
                    triple.split(" | ")[2].strip(),
                )
                # hypothesis = f"The {key} is {value}."
                hypothesis = f"{key}: {value}."
                premise_hypothesises.append((pred, hypothesis))
                hypothesises.append(hypothesis)

            # Check for omissions
            has_omission = False
            scores = model.predict(premise_hypothesises, show_progress_bar=False)
            # (Contradict, Entail, Neutral)
            scores = scores[:, (0, 1)]
            label_mapping = ["C", "E"]
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
            # Compute softmax probabilities
            omission_probs = self.softmax(scores)
            results["mnli.omission_probs"].append(omission_probs[:, 1].tolist())
            if any([label == "C" for label in labels]):
                # If there is a contradiction or a neutral label, we consider the
                # prediction have omissions
                results["mnli.omission"].append(1)
                has_omission = True
            else:
                results["mnli.omission"].append(0)

            # Check for hallucinations
            has_hallucination = False
            premise = " ".join(hypothesises)
            hypothesis = pred
            # (C, E, M)
            scores = model.predict([(premise, hypothesis)], show_progress_bar=False)
            scores = scores[:, (0, 1)]
            # Convert scores to labels (C: contradiction, E: entailment, N: neutral)
            label_mapping = ["C", "E"]
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
            # Compute softmax probabilities
            hallucination_probs = self.softmax(scores)
            results["mnli.hallucination_probs"].append(
                hallucination_probs[:, 1].tolist()
            )
            if any([label == "C" for label in labels]):
                # If there is a contradiction or a neutral label, we consider the
                # prediction to have hallucinations
                results["mnli.hallucination"].append(1)
                has_hallucination = True
            else:
                results["mnli.hallucination"].append(0)

            if has_omission and has_hallucination:
                results["mnli.omission_hallucination"].append(1)
            else:
                results["mnli.omission_hallucination"].append(0)

            if not has_omission and not has_hallucination:
                results["mnli.ok"].append(1)
            else:
                results["mnli.ok"].append(0)

            # Compute the MNLI score as the minimum entailment score for both
            # omissions and hallucinations
            results["mnli.score"].append(
                min(omission_probs[:, 1].tolist() + hallucination_probs[:, 1].tolist())
            )

        return results

    def return_average(self, dict_results):
        results = {
            k: np.mean(v)
            for k, v in dict_results.items()
            if k
            in [
                "mnli.ok",
                "mnli.omission",
                "mnli.hallucination",
                "mnli.omission_hallucination",
                "mnli.score",
            ]
        }
        return results


class SentenceMNLIScorer(BaseScorer):
    fields = ["mnli.sent_hallucination", "mnli.sent_mask"]

    def __init__(self, mnli_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if mnli_path is None:
            self.mnli_path = "cross-encoder/nli-deberta-v3-large"
        else:
            self.mnli_path = mnli_path

    def score(self, dataset):
        model = CrossEncoder(self.mnli_path)
        results = defaultdict(list)
        for triples, pred in zip(dataset["source"], dataset["prediction"]):
            template = ""
            for triple in triples:
                key, value = (
                    triple.split(" | ")[1].strip(),
                    triple.split(" | ")[2].strip(),
                )
                # template += f"The {key} is {value}. "
                template += f"{key}: {value}. "

            # Check for hallucinations
            premise = template.rstrip()
            sentences = sent_tokenize(pred)
            if sentences == []:
                sentences = [""]
            scores = model.predict(
                [(premise, hypothesis) for hypothesis in sentences],
                show_progress_bar=False,
            )
            scores = scores[:, (0, 1)]
            # Convert scores to labels (C: contradiction, E: entailment, N: neutral)
            label_mapping = ["C", "E"]
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

            sent_hallucinated = 0
            sent_mask = []
            for label in labels:
                if label == "C":
                    sent_hallucinated += 1
                    sent_mask.append(1)
                else:
                    sent_mask.append(0)

            results["mnli.sent_hallucination"].append(sent_hallucinated / len(labels))

        return results

    def return_average(self, dict_results):
        results = {
            k: np.mean(v)
            for k, v in dict_results.items()
            if k == "mnli.sent_hallucination"
        }
        return results


class NGramRepetitionScorer(BaseScorer):
    fields = ["repetition.1_gram_rep", "repetition.2_gram_rep"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def count_repetition(self, tokens, n=2):
        ngrams_tokens = ngrams(tokens, n=n)
        fd = FreqDist(ngrams_tokens)
        return 1.0 - float(fd.B()) / max(1, float(fd.N()))

    def ngram_repetition(self, list_inputs, list_n=[1, 2]):
        dict_results = {f"repetition.{n}_gram_rep": [] for n in list_n}
        for text in list_inputs:
            tokens = word_tokenize(text)
            for n in list_n:
                repetitions = self.count_repetition(tokens, n=n)
                dict_results[f"repetition.{n}_gram_rep"].append(repetitions)
        return dict_results

    def score(self, dataset):
        results = self.ngram_repetition(dataset["prediction"])

        return results


class ScorersCollection(BaseScorer):
    def __init__(self, scorer_names: List[str], **kwargs):
        self.list_scorers = self.get_scorers(scorer_names, **kwargs)

    def get_scorers(self, scorers_names: List[str], **kwargs):
        list_scorers = []
        for name in scorers_names:
            if name == "rouge":
                list_scorers.append(RougeScorer(**kwargs))
            elif name == "dataquest":
                list_scorers.append(DataQuestEvalScorer(**kwargs))
            elif name == "bert":
                list_scorers.append(BertScorer(**kwargs))
            elif name == "sacrebleu":
                list_scorers.append(SacreBLEUScorer(**kwargs))
            elif name == "parent-table":
                list_scorers.append(ParentScorer(data_format="table", **kwargs))
            elif name == "parent-graph":
                list_scorers.append(ParentScorer(data_format="graph", **kwargs))
            elif name == "mnli":
                list_scorers.append(MNLIScorer(**kwargs))
            elif name == "mnli-sent":
                list_scorers.append(SentenceMNLIScorer(**kwargs))
            elif name == "repetition":
                list_scorers.append(NGramRepetitionScorer(**kwargs))
            elif name == "bart":
                list_scorers.append(BARTScorer(**kwargs))
            elif name == "align":
                list_scorers.append(AlignScorer(**kwargs))
            else:
                raise ValueError(f"Unknown scorer: {name}")
        return list_scorers

    def score(self, dataset):
        results = {}
        for scorer in self.list_scorers:
            start_time = time.time()
            logging.info(f"Scoring with {scorer.__class__.__name__}...")
            results.update(scorer.score(dataset))
            logging.info(
                f"Scorer {scorer.__class__.__name__} took {time.time() - start_time}s"
            )
        return results

    def return_average(self, dict_results):
        results = {}
        for scorer in self.list_scorers:
            results.update(scorer.return_average(dict_results))
        return results


class DatasetScorer:
    def __init__(
        self,
        file: Any,
        scorers_names: List[str],
        file_type: str = "dataset",
        offline_setup: bool = False,
        **kwargs,
    ):
        self.file: Any = file
        self.file_type: str = file_type

        self.scorers_names = scorers_names
        self.scorers_collection = ScorersCollection(
            scorer_names=self.scorers_names, **kwargs
        )
        self.setup(offline_setup=offline_setup)

    def load_file(self):
        if self.file_type == "csv":
            self.dataset = load_dataset(
                "csv", data_files=str(self.file_path), nrows=self.nrows
            )
        elif self.file_type == "dataset":
            self.dataset = self.file
        self.valid_dataset = self.dataset.filter(lambda x: x["source"] != [])
        print(f"Dataset loaded with {len(self.valid_dataset)} samples")

    def setup(self, offline_setup: bool = False):
        if offline_setup:
            for scorer in self.scorers_collection.list_scorers:
                scorer.offline_setup()
        else:
            logging.info("Loading dataset...")
            self.load_file()

    def score_dataset(self):
        results = self.scorers_collection.score(dataset=self.valid_dataset)
        dataset_results = self.valid_dataset
        for k, v in results.items():
            dataset_results = dataset_results.add_column(k, v)
        return dataset_results

    def get_average_results(self, dataset_results):
        results = self.scorers_collection.return_average(dataset_results)
        return results

    def get_writable_columns(self, dataset_results):
        averaged_results = self.get_average_results(dataset_results)
        columns = list(averaged_results.keys())
        columns += ["context", "prediction", "first_reference", "prompt"]
        return {k: dataset_results[k] for k in columns}


class GPTEvalScorer(BaseScorer):
    """
    Scorer for evaluating the quality of generated text using the GPT-3 model.
    """

    fields = ["Faithfulness", "Non-Hallucination", "Fluency", "Coherence"]

    def __init__(
        self,
        indices,
        model_path,
        oai_key=None,
        oai_endpoint=None,
        oai_model=None,
        api_type=None,
        api_version=None,
        **kwargs,
    ):
        azure_oai_key = (
            oai_key if oai_key is not None else "44de1f18eddc4bf3b18027603025f2f8"
        )
        azure_oai_endpoint = (
            oai_endpoint
            if oai_endpoint is not None
            else "https://air-data2text.openai.azure.com/"
        )
        # Set OpenAI configuration settings
        openai.api_type = api_type if api_type is not None else "azure"
        openai.api_base = azure_oai_endpoint
        openai.api_version = (
            api_version if api_version is not None else "2023-03-15-preview"
        )
        openai.api_key = azure_oai_key
        self.azure_oai_model = (
            oai_model if oai_model is not None else "air-data2text-gpt35-turbo-16k"
        )  # try 'air-data2text' if it doesn't work
        self.indices = indices
        self.model_path = model_path

    def score(self):
        print("Load dataset")
        dataset = load_dataset(
            "csv",
            data_files=str(Path(self.model_path) / "predictions_and_scores.csv"),
        )["train"]

        print("Setup")
        results = defaultdict(list)
        print("Iterating")
        for i in self.indices:
            print("\n", i)
            source = dataset[i]["linearized_input"]
            pred = dataset[i]["prediction"]
            received = False
            system_prompt = """
            Imagine you are a human annotator tasked with evaluating descriptions generated from structured data. Follow these steps:
            1. Analyze the structured data carefully, noting the information it contains.
            2. Read the description based on this data.
            3. Rate the description across four dimensions: faithfulness, non-hallucination, fluency, and coherence, using a 1 to 5 scale. Higher scores always indicate better quality.

            Rating Definitions:
            - Faithfulness: Measures the accuracy of the description against the structured data. Rate 5 for perfectly accurate, 1 for completely inaccurate.
            - Non-Hallucination: Measures if the description avoids adding information not in the data. Rate 5 for no added information, 1 for a lot of added information.
            - Fluency: Assesses the grammatical and stylistic quality of the text. Rate 5 for excellent grammar and style, 1 for poor grammar and style.
            - Coherence: Evaluates how well the sentences fit together logically and naturally. Rate 5 for excellent coherence, 1 for disjointed or illogical text.
            """
            prompt = f"""Here is the structured data and its corresponding description:
            Structured data: {source}
            Description: {pred}.
            Rate the description in the following format: 
            Faithfulness:
            Non-Hallucination:
            Fluency:
            Coherence:
            """
            while not received:
                try:
                    raw_answer = openai.ChatCompletion.create(
                        engine=self.azure_oai_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=32,
                        temperature=0.0,
                        top_p=1.0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                    )
                    answer = raw_answer["choices"][0]["message"]["content"]
                    print("\n" + answer)
                    received = True
                    import re

                    # Define a regular expression pattern to match the criteria and their ratings
                    pattern = r"([A-Za-z]+):\s+(\d+)"

                    # Use the findall method to extract all matches
                    matches = re.findall(pattern, answer)

                    # Iterate through the matches and store them in the dictionary
                    for match in matches:
                        criteria, rating = match
                        results[criteria].append(int(rating))

                except Exception as exc:
                    print(exc)
                    time.sleep(0.1)
            time.sleep(0.1)

        return results
