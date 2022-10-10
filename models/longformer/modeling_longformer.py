# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""PyTorch Longformer model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerModel, \
    LongformerLMHead, LongformerMaskedLMOutput
from transformers.activations import gelu

from transformers.modeling_outputs import (
    ModelOutput,
    SequenceClassifierOutput,
    MultipleChoiceModelOutput
)

@dataclass
class LongformerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LongformerForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of pre-training losses.
        mlm_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The masked language modeling loss.
        srp_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The sentence representation prediction loss.
        drp_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The document representation prediction loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        document_prediction_logits (`torch.FloatTensor` of shape `(batch_size, config.hidden_size)`):
            Prediction scores of the document prediction head (scores for each vocabulary token before Sigmoid).
        sentence_prediction_logits (`torch.FloatTensor` of shape `(batch_size, config.hidden_size)`):
            Prediction scores of the sentence prediction head (scores for each vocabulary token before Sigmoid).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    srp_loss: Optional[torch.FloatTensor] = None
    drp_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    document_prediction_logits: torch.FloatTensor = None
    sentence_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class SentenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
        sentence_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class AttentivePooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_dropout = config.hidden_dropout_prob
        self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)


class LongformerSentenceHead(nn.Module):
    """Hi-Transformer Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.sentence_embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.sentence_embedding_size))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = gelu(features)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class LongformerSentencizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.max_sentence_length = config.max_sentence_length

    def forward(self, hidden_states):
        sentence_repr_hidden_states = hidden_states[:, ::self.max_sentence_length]
        sentence_outputs = self.dense(sentence_repr_hidden_states)
        sentence_outputs = self.activation(sentence_outputs)
        return sentence_outputs


class LongformerPooler(nn.Module):
    def __init__(self, config, pooling='max'):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooling = pooling
        if self.pooling == 'attentive':
            self.attentive_pooling = AttentivePooling(config)
        self.activation = nn.Tanh()
        self.max_sentence_length = config.max_sentence_length

    def forward(self, hidden_states):
        if self.pooling == 'attentive':
            pooled_output = self.attentive_pooling(hidden_states)
        else:
            pooled_output = torch.max(hidden_states, dim=1)[0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerForMaskedLM(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerMaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import LongformerForMaskedLM, LongformerTokenizer

        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        >>> attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
        >>> # check `LongformerModel.forward` for more details how to set *attention_mask*
        >>> outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        >>> loss = outputs.loss
        >>> prediction_logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Init global attention mask
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[input_ids == self.config.sep_token_id] = 1
        global_attention_mask[input_ids == self.config.cls_token_id] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


class LongformerModelForPreTraining(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        if self.config.mlm or self.config.mslm:
            self.lm_head = LongformerLMHead(config)
        if self.config.srp or self.config.srp:
            self.sentencizer = LongformerSentencizer(config)
        if self.config.drp:
            self.pooler = LongformerPooler(config, pooling='max')
            self.document_cls = nn.Linear(config.hidden_size, config.vocab_size)
        if self.config.srp:
            self.sentence_cls = LongformerSentenceHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        document_labels=None,
        sentence_labels=None,
        sentence_masks=None,
        sentence_mask_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Init global attention mask
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[input_ids == self.config.sep_token_id] = 1
        global_attention_mask[input_ids == self.config.cls_token_id] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=None,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Collect sequence output representations
        sequence_output = outputs[0]

        # MLM
        prediction_scores = None
        if self.config.mlm or self.config.mslm:
            prediction_scores = self.lm_head(sequence_output)

        if self.config.srp or self.config.drp:
            sentence_outputs = self.sentencizer(sequence_output)

        # SRP
        sentence_prediction_scores = None
        if self.config.srp:
            sentence_prediction_scores = self.sentence_cls(sentence_outputs)
            if sentence_mask_ids is not None:
                sentence_prediction_scores = sentence_prediction_scores[:, :, sentence_mask_ids].clone()

        # DRP
        document_prediction_scores = None
        if self.config.drp:
            pooled_outputs = self.pooler(sentence_outputs)
            document_prediction_scores = self.document_cls(pooled_outputs)

        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss.clone()

        drp_loss = None
        if document_labels is not None:
            loss_fct = BCEWithLogitsLoss()
            drp_loss = loss_fct(document_prediction_scores, document_labels)
            if labels is not None:
                total_loss += drp_loss
            else:
                total_loss = drp_loss

        srp_loss = None
        if sentence_labels is not None:
            if self.config.sentence_embedding_size != self.config.vocab_size:
                loss_fct = CosineEmbeddingLoss()
                srp_loss = loss_fct(sentence_prediction_scores.view(-1, sentence_labels.shape[-1])[sentence_masks.view(-1).bool()],
                                    sentence_labels.view(-1, sentence_labels.shape[-1])[sentence_masks.view(-1).bool()],
                                    torch.ones((sentence_masks.view(-1).sum(), ), device=sentence_masks.device))
            else:
                loss_fct = BCEWithLogitsLoss()
                srp_loss = loss_fct(sentence_prediction_scores.view(-1, sentence_labels.shape[-1])[sentence_masks.view(-1).bool()],
                                    sentence_labels.view(-1, sentence_labels.shape[-1])[sentence_masks.view(-1).bool()])
            if labels is not None or document_labels is not None:
                total_loss += srp_loss
            else:
                total_loss = srp_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss, masked_lm_loss, srp_loss, drp_loss) + output) if total_loss is not None else output

        return LongformerForPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            srp_loss=srp_loss,
            drp_loss=drp_loss,
            prediction_logits=prediction_scores,
            document_prediction_logits=document_prediction_scores,
            sentence_prediction_logits=sentence_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LongformerModelForSentenceClassification(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.sentencizer = LongformerSentencizer(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Init global attention mask
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[input_ids == self.config.sep_token_id] = 1
        global_attention_mask[input_ids == self.config.cls_token_id] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sentence_outputs = self.sentencizer(sequence_output)
        sentence_outputs = self.dropout(sentence_outputs)
        logits = self.classifier(sentence_outputs)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.view(-1, 1).squeeze(), labels.view(-1).squeeze())
                else:
                    loss = loss_fct(logits.view(-1, 1), labels.view(-1))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                mask = labels[:, :, 0] != -1
                loss = loss_fct(logits[mask], labels[mask])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SentenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sentence_attentions=outputs.sentence_attentions
        )


class LongformerModelForSequenceClassification(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling='max'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.pooling = pooling

        self.longformer = LongformerModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if self.pooling not in ['first', 'last']:
            self.sentencizer = LongformerSentencizer(config)
        self.pooler = LongformerPooler(config, pooling=pooling)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Init global attention mask
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[input_ids == self.config.sep_token_id] = 1
        global_attention_mask[input_ids == self.config.cls_token_id] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.pooling not in ['first', 'last']:
            sentence_outputs = self.sentencizer(sequence_output)
            pooled_output = self.pooler(sentence_outputs)
        elif self.pooling == 'first':
            pooled_output = self.pooler(torch.unsqueeze(sequence_output[:, 0, :], 1))
        elif self.pooling == 'last':
            pooled_output = self.pooler(torch.unsqueeze(sequence_output[:, -128, :], 1))

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LongformerForMultipleChoice(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling='last'):
        super().__init__(config)

        self.pooling = pooling
        self.longformer = LongformerModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if self.pooling not in ['first', 'last']:
            self.sentencizer = LongformerSentencizer(config)
        self.pooler = LongformerPooler(config, pooling=pooling)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()


    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # Init global attention mask
        flat_global_attention_mask = torch.zeros_like(flat_input_ids, device=input_ids.device)
        flat_global_attention_mask[flat_input_ids == self.config.sep_token_id] = 1
        flat_global_attention_mask[flat_input_ids == self.config.cls_token_id] = 1

        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.pooling == 'first':
            pooled_output = self.pooler(torch.unsqueeze(sequence_output[:, 0, :], 1))
        elif self.pooling == 'last':
            pooled_output = self.pooler(torch.unsqueeze(sequence_output[:, -128, :], 1))
        else:
            pooled_output = self.pooler(self.sentencizer(sequence_output))

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
