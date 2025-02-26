from dataclasses import dataclass
from typing import Optional

from transformers import PreTrainedTokenizerBase

from scope.tasks import BaseTask


@dataclass
class LLama2PromptDataCollator:
    tokenizer: PreTrainedTokenizerBase
    task_formater: BaseTask
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, batch, return_tensors=None):
        formatted_batch = self.task_formater.prepare_batch(batch)
        if return_tensors is None:
            return_tensors = self.return_tensors

        tensor_batch = []
        for x in formatted_batch:
            input_ids = self.tokenizer.encode(
                x,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )
            if self.tokenizer.bos_token is not None:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            tensor_batch.append({"input_ids": input_ids})

        features = self.tokenizer.pad(
            tensor_batch,
            return_tensors=return_tensors,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
        )

        return features


class ExpertGuidedDataCollator:
    def __init__(self, main_data_collator, *args):
        self.main_data_collator = main_data_collator
        self.weak_data_collators = args

    def __call__(self, batch):
        main_input = self.main_data_collator(batch)
        weak_inputs = [
            weak_collator(batch) for weak_collator in self.weak_data_collators
        ]

        return main_input, weak_inputs


class ContrastiveLayerDataCollator:
    def __init__(self, main_data_collator, weak_data_collator):
        self.main_data_collator = main_data_collator
        self.weak_data_collator = weak_data_collator

    def __call__(self, batch):
        main_input = self.main_data_collator(batch)
        weak_inputs = self.weak_data_collator(batch)
        return main_input, weak_inputs
