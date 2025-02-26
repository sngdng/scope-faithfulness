# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/alignment-handbook

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class DataCollatorWithPadding:
    r"""
    DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [
                i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
            ]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p
                for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [
                i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
            ]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p
                for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [
                i
                for i, x in enumerate(rejected_tokens["input_ids"])
                if x == eos_token_id
            ]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p
                for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(
                len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
            )

            # if combined sequence is too long, truncate the prompt
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {
                        k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
                    }
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {
                        k: v[-self.max_prompt_length :]
                        for k, v in prompt_tokens.items()
                    }
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                chosen_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in chosen_tokens.items()
                }
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {
                k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
            }
            rejected_sequence_tokens = {
                k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][
                :
            ]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])

            for k, toks in {
                "real": chosen_sequence_tokens,
                "generated": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            rejected_tokens = self.tokenizer(
                rejected,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(
                self.model, "prepare_decoder_input_ids_from_labels"
            ):
                batch["rejected_decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["rejected_labels"]
                    )
                )
                batch["chosen_decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["chosen_labels"]
                    )
                )

        batch["prompt"] = prompt
        batch["real"] = prompt + chosen
        batch["generated"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (
                        (k.startswith("real"))
                        or (k.startswith("generated"))
                        or ("decoder" in k)
                    ):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["real"]
            rejected = feature["generated"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


@dataclass
class DataCollatorWithPaddingForSFT:
    r"""
    DataCollator class that pads the inputs to the maximum length of the batch for SFT.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen.

        We also create the labels for the chosen responses, which are of length equal to
            the sum of the length of the prompt and the chosen response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [
                i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
            ]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p
                for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen
            eos_indices_chosen = [
                i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
            ]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p
                for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            longer_response_length = len(chosen_tokens["input_ids"])

            # if combined sequence is too long, truncate the prompt
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {
                        k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
                    }
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {
                        k: v[-self.max_prompt_length :]
                        for k, v in prompt_tokens.items()
                    }
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                chosen_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in chosen_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {
                k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])

            batch["input_ids"] = chosen_sequence_tokens["input_ids"]
            batch["attention_mask"] = chosen_sequence_tokens["attention_mask"]
            batch["labels"] = chosen_sequence_tokens["labels"]

        else:
            chosen_tokens = self.tokenizer(
                chosen,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )

            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch["labels"] = chosen_tokens["input_ids"]
            batch["input_ids"] = prompt_tokens["input_ids"]
            batch["attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(
                self.model, "prepare_decoder_input_ids_from_labels"
            ):
                batch["decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["labels"]
                    )
                )

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("input_ids")
                or k.endswith("attention_mask")
                or k.endswith("labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (
                        (k.startswith("real"))
                        or (k.startswith("generated"))
                        or ("decoder" in k)
                    ):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["real"]

            batch_element = self.tokenize_batch_element(prompt, chosen)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)


@dataclass
class DataCollatorWithPaddingForCLIFF:
    r"""
    DataCollator class for CLIFF Trainer that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        bt: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            bt_tokens = self.tokenizer(bt, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [
                i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id
            ]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p
                for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen, bt and rejected
            eos_indices_chosen = [
                i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id
            ]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p
                for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_bt = [
                i for i, x in enumerate(bt_tokens["input_ids"]) if x == eos_token_id
            ]
            new_attention_mask_bt = [
                0 if i in eos_indices_bt else p
                for i, p in enumerate(bt_tokens["attention_mask"])
            ]

            eos_indices_rejected = [
                i
                for i, x in enumerate(rejected_tokens["input_ids"])
                if x == eos_token_id
            ]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p
                for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            bt_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            bt_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(
                len(chosen_tokens["input_ids"]),
                len(bt_tokens["input_ids"]),
                len(rejected_tokens["input_ids"]),
            )

            # if combined sequence is too long, truncate the prompt
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {
                        k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()
                    }
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {
                        k: v[-self.max_prompt_length :]
                        for k, v in prompt_tokens.items()
                    }
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if (
                len(prompt_tokens["input_ids"]) + longer_response_length
                > self.max_length
            ):
                chosen_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in chosen_tokens.items()
                }
                bt_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in bt_tokens.items()
                }
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length]
                    for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {
                k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
            }
            bt_sequence_tokens = {k: prompt_tokens[k] + bt_tokens[k] for k in bt_tokens}
            rejected_sequence_tokens = {
                k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])
            bt_sequence_tokens["labels"] = bt_sequence_tokens["input_ids"][:]
            bt_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][
                :
            ]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
                self.label_pad_token_id
            ] * len(prompt_tokens["input_ids"])

            for k, toks in {
                "real": chosen_sequence_tokens,
                "bt": bt_sequence_tokens,
                "generated": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            bt_tokens = self.tokenizer(
                bt,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            rejected_tokens = self.tokenizer(
                rejected,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["bt_labels"] = bt_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(
                self.model, "prepare_decoder_input_ids_from_labels"
            ):
                batch["rejected_decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["rejected_labels"]
                    )
                )
                batch["chosen_decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["chosen_labels"]
                    )
                )
                batch["bt_decoder_input_ids"] = (
                    self.model.prepare_decoder_input_ids_from_labels(
                        labels=batch["bt_labels"]
                    )
                )

        batch["prompt"] = prompt
        batch["real"] = prompt + chosen
        batch["bt"] = prompt + bt
        batch["generated"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (
                        (k.startswith("real"))
                        or (k.startswith("generated"))
                        or ("decoder" in k)
                    ):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["real"]
            bt = feature["bt"]
            rejected = feature["neg"]

            batch_element = self.tokenize_batch_element(prompt, chosen, bt, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
