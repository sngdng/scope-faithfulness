import math
from typing import Optional

import torch
from transformers import UnbatchedClassifierFreeGuidanceLogitsProcessor


class MixtureLogitsProcessor(UnbatchedClassifierFreeGuidanceLogitsProcessor):
    def __init__(
        self,
        mixture_alpha: float,
        mixture_mode: str,
        unconditional_model: Optional[torch.nn.Module] = None,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        super().__init__(
            model=unconditional_model,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            use_cache=use_cache,
            guidance_scale=mixture_alpha,
        )
        mixture_alpha = float(mixture_alpha)
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        if mixture_alpha != 0.0 and mixture_alpha != 1.0:
            self.log_alpha = math.log(mixture_alpha)
            self.log_minus_alpha = math.log(1.0 - mixture_alpha)
        else:
            self.log_alpha = 0.0
            self.log_minus_alpha = 0.0

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.mixture_alpha == 0.0:
            return scores

        logits = self.get_unconditional_logits(input_ids)[:, -1, :]

        inf_indices = torch.isinf(scores)
        logits[inf_indices] = float("-Inf")

        unconditional_logits = torch.nn.functional.log_softmax(logits, dim=-1)
        if self.mixture_alpha == 1.0:
            return unconditional_logits

        if self.mixture_mode == "unfaithful":
            if torch.rand(1).item() >= self.mixture_alpha:
                scores_processed = scores
            else:
                scores_processed = unconditional_logits

        elif self.mixture_mode == "cad":
            unconditional_indices = torch.isinf(unconditional_logits)
            unconditional_logits[unconditional_indices] = float("Inf")
            scores_processed = (
                1.0 + self.mixture_alpha
            ) * scores - self.mixture_alpha * unconditional_logits
        else:
            raise ValueError(f"Invalid mixture mode: {self.mixture_mode}")

        return scores_processed
