import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    BeamScorer,
    LogitsProcessorList,
    PretrainedConfig,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation import LogitsProcessor, TopKLogitsWarper
from transformers.generation.utils import (
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from scope.logits_processor import MixtureLogitsProcessor


class MixtureDecoder(nn.Module):
    def __init__(self, model, unconditional_model, mixture_alpha, mixture_mode):
        super().__init__()
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        self.model = model

    def generate(self, **kwargs):
        weak_inputs = kwargs.pop("weak_inputs")

        processor = MixtureLogitsProcessor(
            unconditional_model=self.unconditional_model,
            unconditional_ids=weak_inputs[0]["input_ids"],
            unconditional_attention_mask=weak_inputs[0]["attention_mask"],
            mixture_mode=self.mixture_mode,
            mixture_alpha=self.mixture_alpha,
        )

        processor = LogitsProcessorList([processor])

        return self.model.generate(logits_processor=processor, **kwargs)

    # method as attribute
    @property
    def device(self):
        return self.model.device

    def compute_transition_scores(self, *args, **kwargs):
        return self.model.compute_transition_scores(*args, **kwargs)


class BaseExpertGuidedDecoder(PreTrainedModel):
    def __init__(self, main_model, *args, **kwargs):
        super().__init__(**kwargs)
        self.main_model = main_model
        self.weak_models = nn.ModuleList(args)

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(
                    dict_to_expand[key], torch.Tensor
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(
                        expand_size, dim=0
                    )
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        new_weak_inputs = []

        for weak in model_kwargs["weak_inputs"]:
            new_input = _expand_dict_for_generation(weak)
            new_weak_inputs.append(new_input)
        model_kwargs["weak_inputs"] = new_weak_inputs

        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask,
        weak_inputs,
        past_key_values=None,
        **kwargs,
    ):
        main_inputs = self.main_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if "use_cache" in kwargs.keys():
            for i in range(len(weak_inputs)):
                weak_inputs[i]["use_cache"] = True

        new_weak_inputs = []
        for i, (weak_model, weak_input) in enumerate(
            zip(self.weak_models, weak_inputs)
        ):
            new_weak_inputs.append(
                weak_model.prepare_inputs_for_generation(**weak_input)
            )
        model_inputs = main_inputs
        model_inputs["weak_inputs"] = new_weak_inputs
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        weak_outputs = outputs.weak_outputs
        # Assuming weak_outputs and weak_inputs are aligned and have the same structure
        for i, weak_output in enumerate(weak_outputs):
            weak_model_input = model_kwargs["weak_inputs"][i]
            weak_model_input["past_key_values"] = self._extract_past_from_model_output(
                weak_output, standardize_cache_format=standardize_cache_format
            )

            # Update token_type_ids for weak input, if applicable
            if "token_type_ids" in weak_model_input:
                token_type_ids = weak_model_input["token_type_ids"]
                weak_model_input["token_type_ids"] = torch.cat(
                    [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
                )

            # Update attention masks for weak input
            if not is_encoder_decoder:
                # Update attention mask for a non-encoder-decoder model
                if "attention_mask" in weak_model_input:
                    attention_mask = weak_model_input["attention_mask"]
                    weak_model_input["attention_mask"] = torch.cat(
                        [
                            attention_mask,
                            attention_mask.new_ones((attention_mask.shape[0], 1)),
                        ],
                        dim=-1,
                    )
            else:
                # Update decoder attention mask for an encoder-decoder model
                if "decoder_attention_mask" in weak_model_input:
                    decoder_attention_mask = weak_model_input["decoder_attention_mask"]
                    weak_model_input["decoder_attention_mask"] = torch.cat(
                        [
                            decoder_attention_mask,
                            decoder_attention_mask.new_ones(
                                (decoder_attention_mask.shape[0], 1)
                            ),
                        ],
                        dim=-1,
                    )

        return model_kwargs

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            raise AttributeError(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )

        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else self.generation_config.pad_token_id
        )
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.generation_config.eos_token_id
        )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = (
            output_scores
            if output_scores is not None
            else self.generation_config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size))
            if (return_dict_in_generate and output_scores)
            else None
        )
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = next_token_logits
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                :, None
            ].expand_as(next_token_scores_processed)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores,
                max(2, 1 + n_eos_tokens) * num_beams,
                dim=1,
                largest=True,
                sorted=True,
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            for model_input in model_kwargs["weak_inputs"]:
                model_input["input_ids"] = torch.cat(
                    [
                        model_input["input_ids"][beam_idx, :],
                        beam_next_tokens.unsqueeze(-1),
                    ],
                    dim=-1,
                )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.main_model._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            for i, weak_input in enumerate(model_kwargs["weak_inputs"]):
                if weak_input["past_key_values"] is not None:
                    model_kwargs["weak_inputs"][i]["past_key_values"] = (
                        self.weak_models[i]._reorder_cache(
                            weak_input["past_key_values"], beam_idx
                        )
                    )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple(
                    (
                        beam_indices[beam_idx[i]] + (beam_idx[i],)
                        for i in range(len(beam_indices))
                    )
                )

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        weak_inputs=None,
        **kwargs,
    ):
        main_output = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        weak_outputs = [
            weak_model(**weak_input)
            for weak_input, weak_model in zip(weak_inputs, self.weak_models)
        ]

        processed_logits = main_output.logits
        processed_weak_logits = [weak_out.logits for weak_out in weak_outputs]
        ## generation https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/generation/utils.py#L3103 log sfotmax is applied after in beam search
        processed_scores = nn.functional.log_softmax(processed_logits, dim=-1)
        processed_weak_scores = [
            nn.functional.log_softmax(weak_logits, dim=-1)
            for weak_logits in processed_weak_logits
        ]
        return BaseExpertGuidedDecoderOutput(
            main_output=main_output,
            weak_outputs=weak_outputs,
            processed_scores=processed_scores,
            processed_weak_scores=processed_weak_scores,
        )


@dataclass
class BaseExpertGuidedDecoderOutput:
    main_output: Any
    weak_outputs: List[Any]
    processed_scores: torch.FloatTensor
    processed_weak_scores: List[torch.FloatTensor]


@dataclass
class ExpertGuidedCausalLMOutput(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    weak_outputs: Optional[Tuple[torch.FloatTensor]] = None


class ContrastiveDecoder(BaseExpertGuidedDecoder):
    def __init__(self, main_model, *args, **kwargs):
        super().__init__(main_model, *args, **kwargs)

        self.alpha = self.config.alpha

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        weak_inputs=None,
        **kwargs,
    ):
        models_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            weak_inputs=weak_inputs,
        )
        processed_scores = models_outputs.processed_scores
        processed_weak_scores = models_outputs.processed_weak_scores

        if self.alpha is not None:
            if len(processed_weak_scores) > 1:
                print(
                    "Warning: the number of weak models logits is greater than 1, using the first one for alpha."
                )
            max_confidence = torch.max(processed_scores, dim=-1, keepdim=True)
            indices_to_remove = (
                processed_scores
                < torch.log(torch.tensor(self.alpha)) + max_confidence.values
            )
        else:
            indices_to_remove = torch.zeros_like(processed_scores, dtype=torch.bool)

        processed_scores = processed_scores.masked_fill_(
            indices_to_remove, -float("Inf")
        )

        for weak_scores in processed_weak_scores:
            processed_scores[:, -1, :] = processed_scores[:, -1, :].sub_(
                weak_scores[:, -1, :]
            )
        weak_outputs = [
            ExpertGuidedCausalLMOutput(
                loss=weak_output.loss,
                logits=weak_output.logits,
                past_key_values=weak_output.past_key_values,
            )
            for weak_output in models_outputs.weak_outputs
        ]

        return ExpertGuidedCausalLMOutput(
            loss=models_outputs.main_output.loss,
            logits=processed_scores,
            past_key_values=models_outputs.main_output.past_key_values,
            weak_outputs=weak_outputs,
        )


class ContextAwareDecoder(BaseExpertGuidedDecoder):
    def __init__(self, main_model, *args, **kwargs):
        super().__init__(main_model, *args, **kwargs)
        self.alpha = self.config.alpha

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        weak_inputs=None,
        **kwargs,
    ):
        models_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            weak_inputs=weak_inputs,
        )
        processed_scores = models_outputs.processed_scores
        processed_weak_scores = models_outputs.processed_weak_scores

        for weak_scores in processed_weak_scores:
            if self.alpha is not None:
                processed_scores[:, -1, :] = (
                    (1.0 + self.alpha) * processed_scores[:, -1, :]
                ).sub_(self.alpha * weak_scores[:, -1, :])

        weak_outputs = [
            ExpertGuidedCausalLMOutput(
                loss=weak_output.loss,
                logits=weak_output.logits,
                past_key_values=weak_output.past_key_values,
            )
            for weak_output in models_outputs.weak_outputs
        ]

        return ExpertGuidedCausalLMOutput(
            loss=models_outputs.main_output.loss,
            logits=processed_scores,
            past_key_values=models_outputs.main_output.past_key_values,
            weak_outputs=weak_outputs,
        )


class PMIDecoder(BaseExpertGuidedDecoder):
    def __init__(self, main_model, *args, **kwargs):
        super().__init__(main_model, *args, **kwargs)
        self.tau = self.config.tau
        self.lambd = self.config.lambd

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        weak_inputs=None,
        **kwargs,
    ):
        models_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            weak_inputs=weak_inputs,
        )
        processed_scores = models_outputs.processed_scores
        processed_weak_scores = models_outputs.processed_weak_scores

        # Copy from HuggingFace TopKLogitsWrapper
        next_token_entropy = -(
            torch.exp(processed_scores[:, -1, :]) * processed_scores[:, -1, :]
        ).sum(dim=-1, keepdim=True)

        indices_to_remove = (next_token_entropy >= self.tau).float()

        for weak_scores in processed_weak_scores:
            processed_scores[:, -1, :] = processed_scores[:, -1, :].sub_(
                self.lambd * indices_to_remove * weak_scores[:, -1, :]
            )

        weak_outputs = [
            ExpertGuidedCausalLMOutput(
                loss=weak_output.loss,
                logits=weak_output.logits,
                past_key_values=weak_output.past_key_values,
            )
            for weak_output in models_outputs.weak_outputs
        ]

        return ExpertGuidedCausalLMOutput(
            loss=models_outputs.main_output.loss,
            logits=processed_scores,
            past_key_values=models_outputs.main_output.past_key_values,
            weak_outputs=weak_outputs,
        )


class GeneralCriticLogitsWarper(LogitsProcessor):
    def __init__(
        self,
        model,
        main_tokenizer,
        conditioning_model,
        conditioning_tokenizer,
        top_k: int,
        filter_value: float = -float("Inf"),
        beam_size: int = 1,
        condition_lambda=1.0,
        max_length=4096,
        linear_warnup=True,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(
                f"`top_k` has to be a strictly positive integer, but is {top_k}"
            )
        self.conditioning_tokenizer = conditioning_tokenizer
        self.conditioning_model = conditioning_model
        self.model = model
        self.main_tokenizer = main_tokenizer
        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = 1
        self.beam_size = beam_size
        self.condition_lambda = condition_lambda
        self.max_length = max_length
        self.linear_warmup = linear_warnup

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        top_k = min(
            max(self.top_k, self.min_tokens_to_keep), scores.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        topk_res = torch.topk(scores, top_k)

        # merge input_ids to token_ids
        new_tokens = self.main_tokenizer.batch_decode(
            topk_res.indices.reshape(-1),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        new_prefixes = self.main_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        premises = []
        for i, prefix in enumerate(new_prefixes):
            for j in range(top_k):
                premises.append(prefix + new_tokens[i * top_k + j])

        # run conditional model
        cond_logits = self._run_conditional_model(premises, new_prefixes)

        # merge logits
        cond_logits = (
            cond_logits.reshape((-1, top_k)) * self.condition_lambda
        )  # HERE condition_lambda=1.0
        words_in_premise = len(premises[0].strip().split(" "))
        if self.linear_warmup is True and words_in_premise < 6:
            if words_in_premise < 2:
                cond_logits = cond_logits * 0.0
            else:
                cond_logits = cond_logits * ((words_in_premise - 1) / 5.0)
        cond_logits = cond_logits.type(scores.dtype)  # needed for 8bit
        scores = scores.scatter(1, topk_res.indices, cond_logits, reduce="add")
        return scores


class ClassifierCriticLogitsWarper(GeneralCriticLogitsWarper):
    """
    String Matcher
    """

    def __init__(
        self,
        model,
        main_tokenizer,
        conditioning_model,
        conditioning_tokenizer,
        top_k: int,
        filter_value: float = -float("Inf"),
        beam_size=1,
        condition_lambda=1.0,
        max_length=4096,
        linear_warmup=True,
    ):
        super().__init__(
            model,
            main_tokenizer,
            conditioning_model,
            conditioning_tokenizer,
            top_k,
            filter_value,
            beam_size,
            condition_lambda,
            max_length,
            linear_warmup,
        )

    def _run_conditional_model(self, premises, inputs):
        hypothesies = list(
            itertools.chain.from_iterable(
                itertools.repeat(i, self.top_k * self.beam_size) for i in inputs
            )
        )
        cond_features1 = self.conditioning_tokenizer(
            hypothesies,
            premises,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to("cuda")
        cond_logits = self.conditioning_model._forward(cond_features1)
        return cond_logits


class CriticDecoder(PreTrainedModel):
    def __init__(
        self,
        main_model,
        main_tokenizer,
        critic_model,
        critic_tokenizer,
        max_length,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.main_model = main_model

        self.lambd = self.config.lambd
        self.critic_top_k = self.config.critic_top_k
        self.linear_warmup = self.config.linear_warmup
        self.critic_model = critic_model

        self.wrapper = ClassifierCriticLogitsWarper(
            model=self.main_model,
            conditioning_model=self.critic_model,
            main_tokenizer=main_tokenizer,
            conditioning_tokenizer=critic_tokenizer,
            max_length=max_length,
            top_k=self.critic_top_k,
            beam_size=self.config.num_beams,
            condition_lambda=self.lambd,
            linear_warmup=self.linear_warmup,
        )
        self.processors = LogitsProcessorList()
        self.processors.append(TopKLogitsWarper(top_k=self.critic_top_k))
        self.processors.append(self.wrapper)

    def generate(self, *args, **kwargs):
        return self.main_model.generate(
            *args, **kwargs, logits_processor=self.processors
        )


class BaseExpertGuidedDecoderConfig(PretrainedConfig):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.is_encoder_decoder = False
        self.vocab_size = vocab_size


class ContrastiveDecoderConfig(BaseExpertGuidedDecoderConfig):
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.candidate_layers = "all"


class ContextAwareDecoderConfig(BaseExpertGuidedDecoderConfig):
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha


class PMIDecoderConfig(BaseExpertGuidedDecoderConfig):
    def __init__(self, lambd=0.066, tau=3.6, **kwargs):
        super().__init__(**kwargs)
        self.lambd = lambd
        self.tau = tau


class CriticDecoderConfig(BaseExpertGuidedDecoderConfig):
    def __init__(
        self,
        lambd=0.25,
        critic_top_k=5,
        linear_warmup=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambd = lambd
        self.critic_top_k = critic_top_k
        self.linear_warmup = linear_warmup
