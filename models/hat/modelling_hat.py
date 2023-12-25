# coding=utf-8
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
"""PyTorch HAT model."""

import torch
import torch.utils.checkpoint
from packaging import version
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from torch.nn.functional import normalize

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    ModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaIntermediate, RobertaOutput
from transformers.activations import gelu
from transformers import PretrainedConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "kiddothe2b/hierarchical-transformer-base-4096"
_CONFIG_FOR_DOC = "HATConfig"
_TOKENIZER_FOR_DOC = "HATTokenizer"

HAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kiddothe2b/hierarchical-transformer-base-4096",
    "kiddothe2b/adhoc-hierarchical-transformer-base-4096",
    # See all HAT models at https://huggingface.co/models?filter=hierarchical-transformer
]


def transform_tokens2sentences(hidden_states, num_sentences, max_sentence_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), num_sentences, max_sentence_length, hidden_states.size(-1)))
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_sentences,
                                                                max_sentence_length, seg_hidden_states.size(-1))

    return hidden_states_reshape


def transform_masks2sentences(hidden_states, num_sentences, max_sentence_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), 1, 1, num_sentences, max_sentence_length))
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_sentences,
                                                                1, 1, seg_hidden_states.size(-1))

    return hidden_states_reshape


def transform_sentences2tokens(seg_hidden_states, num_sentences, max_sentence_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(seg_hidden_states.size(0) // num_sentences, num_sentences,
                                                        max_sentence_length, seg_hidden_states.size(-1))
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(hidden_states.size(0), num_sentences * max_sentence_length,
                                                    hidden_states.size(-1))
    return hidden_states


@dataclass
class BaseModelOutputWithSentenceAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        sentence_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Sentence attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    sentence_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceRepresentationOutput(ModelOutput):
    """
    Base class for outputs of document representation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        representations (`torch.FloatTensor` of shape `(batch_size, config.hidden_size)`):
            Latent representations.
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
    representations: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class HATForBoWPreTrainingOutput(ModelOutput):
    """
    Output type of [`HATForPreTraining`].

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
class HATForVICRegPreTrainingOutput(ModelOutput):
    """
    Output type of [`HATForVICRegPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of pre-training losses.
        mlm_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The masked language modeling loss.
        sent_sim_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The sentence similarity loss.
        doc_sim_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The document similarity loss.
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
    sent_sim_loss: Optional[torch.FloatTensor] = None
    sent_std_loss: Optional[torch.FloatTensor] = None
    sent_cov_loss: Optional[torch.FloatTensor] = None
    pre_sent_std_loss: Optional[torch.FloatTensor] = None
    pre_sent_cov_loss: Optional[torch.FloatTensor] = None
    doc_sim_loss: Optional[torch.FloatTensor] = None
    doc_std_loss: Optional[torch.FloatTensor] = None
    doc_cov_loss: Optional[torch.FloatTensor] = None
    pre_doc_std_loss: Optional[torch.FloatTensor] = None
    pre_doc_cov_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    document_prediction_logits: torch.FloatTensor = None
    sentence_prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class HATForSimCLRPreTrainingOutput(ModelOutput):
    """
    Output type of [`HATForSimCLRPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of pre-training losses.
        mlm_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The masked language modeling loss.
        sent_sim_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The sentence similarity loss.
        doc_sim_loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        The document similarity loss.
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
    sent_contr_loss: Optional[torch.FloatTensor] = None
    sent_std_loss: Optional[torch.FloatTensor] = None
    sent_cov_loss: Optional[torch.FloatTensor] = None
    doc_contr_loss: Optional[torch.FloatTensor] = None
    doc_std_loss: Optional[torch.FloatTensor] = None
    doc_cov_loss: Optional[torch.FloatTensor] = None
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
    sentence_attentions: Optional[Tuple[torch.FloatTensor]] = None


class HATConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.HAT`.
    It is used to instantiate a HAT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the HAT `kiddothe2b/hat-base-4096 <https://huggingface.co/kiddothe2b/hat-base-4096>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        max_sentences (:obj:`int`, `optional`, defaults to 64):
            The maximum number of sentences that this model might ever be used with.
        max_sentence_size (:obj:`int`, `optional`, defaults to 128):
            The maximum sentence length that this model might ever be used with.
        model_max_length (:obj:`int`, `optional`, defaults to 8192):
            The maximum  sequence length (max_sentences * max_sentence_size) that this model might ever be used with
        encoder_layout (:obj:`Dict`):
            The sentence/document encoder layout.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.
    """
    model_type = "hierarchical-transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        max_sentences=64,
        max_sentence_size=128,
        model_max_length=8192,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        encoder_layout=None,
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_sentences = max_sentences
        self.max_sentence_size = max_sentence_size
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class HATEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_sentence_length + self.padding_idx + 1, config.hidden_size, padding_idx=self.padding_idx)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(self.padding_idx + 1,
                                                          config.max_sentence_length + self.padding_idx + 1).repeat(config.max_sentences).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long),
                persistent=False,
            )

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, self.position_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class HATLayer(nn.Module):
    def __init__(self, config, use_sentence_encoder=True, use_document_encoder=True):
        super().__init__()
        self.max_sentence_length = config.max_sentence_length
        self.max_sentences = config.max_sentences
        self.hidden_size = config.hidden_size
        self.use_document_encoder = use_document_encoder
        self.use_sentence_encoder = use_sentence_encoder
        if self.use_sentence_encoder:
            self.sentence_encoder = TransformerLayer(config)
        if self.use_document_encoder:
            self.document_encoder = TransformerLayer(config)
            self.position_embeddings = nn.Embedding(config.max_sentences+1, config.hidden_size,
                                                    padding_idx=config.pad_token_id)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        output_attentions=False,
    ):

        sentence_outputs = (None, None)
        if self.use_sentence_encoder:
            # transform sequences to sentences
            sentence_inputs = transform_tokens2sentences(hidden_states,
                                                         num_sentences=num_sentences,
                                                         max_sentence_length=self.max_sentence_length)
            sentence_masks = transform_masks2sentences(attention_mask,
                                                       num_sentences=num_sentences,
                                                       max_sentence_length=self.max_sentence_length)

            sentence_outputs = self.sentence_encoder(sentence_inputs,
                                                     sentence_masks,
                                                     output_attentions=output_attentions)

            # transform sentences to tokens
            outputs = transform_sentences2tokens(sentence_outputs[0],
                                                 num_sentences=num_sentences,
                                                 max_sentence_length=self.max_sentence_length)
        else:
            outputs = hidden_states

        document_outputs = (None, None)
        if self.use_document_encoder:
            # gather sentence representative tokens
            sentence_global_tokens = outputs[:, ::self.max_sentence_length].clone()
            sentence_attention_mask = attention_mask[:, :, :, ::self.max_sentence_length].clone()

            sentence_positions = torch.arange(1, num_sentences+1).repeat(outputs.size(0), 1).to(outputs.device) \
                                 * (sentence_attention_mask.reshape(-1, num_sentences) >= -100).int().to(outputs.device)
            # outputs[:, ::self.max_sentence_length] += self.position_embeddings(sentence_positions)
            sentence_global_tokens += self.position_embeddings(sentence_positions)

            document_outputs = self.document_encoder(sentence_global_tokens,
                                                     sentence_attention_mask,
                                                     output_attentions=output_attentions)

            # replace sentence representative tokens
            outputs[:, ::self.max_sentence_length] = document_outputs[0]

        if output_attentions:
            return outputs, sentence_outputs[1], document_outputs[1]

        return outputs, None


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class HATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([HATLayer(config,
                                                       use_sentence_encoder=self.config.encoder_layout[str(idx)]['sentence_encoder'],
                                                       use_document_encoder=self.config.encoder_layout[str(idx)]['document_encoder'])
                                    for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        num_sentences=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_sentence_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    num_sentences,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_sentence_attentions = all_sentence_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_sentence_attentions
                ]
                if v is not None
            )
        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            sentence_attentions=all_sentence_attentions,
        )

    def _tie_weights(self):
        """
        Tie the weights between sentence positional embeddings across all layers.
        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        original_position_embeddings = None
        for module in self.layer:
            if hasattr(module, "position_embeddings"):
                    assert hasattr(module.position_embeddings, "weight")
                    if original_position_embeddings is None:
                        original_position_embeddings = module.position_embeddings
                    if self.config.torchscript:
                        module.position_embeddings.weight = nn.Parameter(original_position_embeddings.weight.clone())
                    else:
                        module.position_embeddings.weight = original_position_embeddings.weight
        return


class HATPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HATConfig
    base_model_prefix = "hat"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HATEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


HAT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`HATConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

HAT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`HATTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


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


class HATPooler(nn.Module):
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


class HATSentencizer(nn.Module):
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

@add_start_docstrings(
    "The bare HAT Model transformer outputting raw hidden-states without any specific head on top.",
    HAT_START_DOCSTRING,
)
class HATModel(HATPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->HAT
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = HATEmbeddings(config)
        self.encoder = HATEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithSentenceAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Compute number of sentences
        num_batch_sentences = input_ids.shape[-1] // self.config.max_sentence_length

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            num_sentences=num_batch_sentences,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            sentence_attentions=encoder_outputs.sentence_attentions,
        )


class HATLMHead(nn.Module):
    """HAT Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class HATSentenceHead(nn.Module):
    """HAT Head for masked language modeling."""

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


class HATSiameseHead(nn.Module):
    """HAT Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False)

    def forward(self, features):
        x = self.dense(features)
        return x


@add_start_docstrings("""HAT Model with a `language modeling` head on top.""", HAT_START_DOCSTRING)
class HATForMaskedLM(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.hi_transformer = HATModel(config)
        self.lm_head = HATLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.hi_transformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hi_transformer.embeddings.word_embeddings = value

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
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

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HATModelForDocumentRepresentation(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling='max'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.max_sentence_length = config.max_sentence_length

        self.hi_transformer = HATModel(config)
        self.pooler = HATPooler(config, pooling=pooling)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        pooled_outputs = self.pooler(sequence_output[:, ::self.max_sentence_length])

        drp_loss = None
        if labels is not None:
            loss_fct = MSELoss()
            drp_loss = loss_fct(pooled_outputs, labels)

        if not return_dict:
            output = (pooled_outputs,) + outputs[2:]
            return ((drp_loss,) + output) if drp_loss is not None else output

        return SequenceRepresentationOutput(
            loss=drp_loss,
            representations=pooled_outputs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(""" HAT Model transformer for masked sentence representation prediction """,
    HAT_START_DOCSTRING,
)
class HATModelForMaskedSentenceRepresentation(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.hi_transformer = HATModel(config)
        self.sentencizer = HATSentencizer(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sentence_outputs = self.sentencizer(sequence_output)

        srp_loss = None
        if labels is not None:
            loss_fct = MSELoss()
            srp_loss = loss_fct(sentence_outputs, labels)

        if not return_dict:
            output = (sentence_outputs,) + outputs[2:]
            return ((srp_loss,) + output) if srp_loss is not None else output

        return SequenceRepresentationOutput(
            loss=srp_loss,
            representations=sentence_outputs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    HAT Model with three heads on top as done during the pretraining: a `masked language modeling` head and a `document
    representation prediction ` head and a `masked sentence representation prediction ` head.
    """,
    HAT_START_DOCSTRING,
)
class HATModelForBoWPreTraining(HATPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.hi_transformer = HATModel(config)
        if self.config.mlm or self.config.mslm:
            self.lm_head = HATLMHead(config)
        if self.config.srp or self.config.srp:
            self.sentencizer = HATSentencizer(config)
        if self.config.drp:
            self.pooler = HATPooler(config, pooling='max')
            self.document_cls = nn.Linear(config.hidden_size, config.vocab_size)
        if self.config.srp:
            self.sentence_cls = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

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
        document_mask_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Collect sequence output representations
        sequence_output = outputs[0]

        # Masked Language Modeling (MLM)
        prediction_scores = None
        if self.config.mlm or self.config.mslm:
            prediction_scores = self.lm_head(sequence_output)

        if self.config.srp or self.config.drp:
            sentence_outputs = self.sentencizer(sequence_output)

        # Sentence Representation Prediction (SRP)
        sentence_prediction_scores = None
        if self.config.srp:
            sentence_prediction_scores = self.sentence_cls(sentence_outputs)
            if sentence_mask_ids is not None:
                sentence_prediction_scores = sentence_prediction_scores[:, :, sentence_mask_ids].clone()

        # Document Representation Prediction (DRP)
        document_prediction_scores = None
        if self.config.drp:
            pooled_outputs = self.pooler(sentence_outputs)
            document_prediction_scores = self.document_cls(pooled_outputs)
            if document_mask_ids is not None:
                document_prediction_scores = document_prediction_scores[:, document_mask_ids].clone()

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

        return HATForBoWPreTrainingOutput(
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


@add_start_docstrings(
    """
    HAT Model with three heads on top as done during the pretraining: a `masked language modeling` head and a `sentence
    projection head` head and a document projection head` head.
    """,
    HAT_START_DOCSTRING,
)
class HATModelForVICRegPreTraining(HATPreTrainedModel):
    def __init__(self, config,
                 document_regularization=True,
                 sentence_regularization=True):
        super().__init__(config)

        self.document_regularization = document_regularization
        self.sentence_regularization = sentence_regularization
        self.hi_transformer = HATModel(config)
        if self.config.mlm:
            self.lm_head = HATLMHead(config)
        if self.config.sent_sim or self.config.doc_sim:
            self.sentencizer = HATSentencizer(config)
            self.cosine = nn.CosineSimilarity(dim=1)
        if self.config.doc_sim:
            self.pooler = HATPooler(config, pooling='max')
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        secondary_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        secondary_labels=None,
        sentence_masks=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        primary_outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        secondary_outputs = self.hi_transformer(
            secondary_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Collect sequence output representations
        primary_sequence_output = primary_outputs[0]
        secondary_sequence_output = secondary_outputs[0]

        # Masked Language Modeling (MLM)
        primary_prediction_scores = None
        secondary_prediction_scores = None
        if self.config.mlm:
            primary_prediction_scores = self.lm_head(primary_sequence_output)
            if secondary_labels is not None:
                secondary_prediction_scores = self.lm_head(secondary_sequence_output)

        if self.config.sent_sim or self.config.doc_sim:
            primary_sentence_outputs = self.sentencizer(primary_sequence_output)
            secondary_sentence_outputs = self.sentencizer(secondary_sequence_output)

        # Document Representation Prediction (DRP)
        if self.config.doc_sim:
            primary_pooled_outputs = self.pooler(primary_sentence_outputs)
            secondary_pooled_outputs = self.pooler(secondary_sentence_outputs)


        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(primary_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss.clone() / 2
        if secondary_labels is not None:
            masked_lm_loss = loss_fct(secondary_prediction_scores.view(-1, self.config.vocab_size), secondary_labels.view(-1))
            total_loss += masked_lm_loss / 2

        sent_sim_loss = None
        sent_std_loss = None
        sent_cov_loss = None
        pre_sent_std_loss = None
        pre_sent_cov_loss = None
        if self.config.sent_sim:
            # sentence projections similarity
            sent_sim_loss = 1 - self.cosine(
                primary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size),
                secondary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size)).mean()
            # sentence projections variance, covariance
            sent_std_loss, sent_cov_loss = vic_reg(
                primary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size),
                secondary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size))

            if labels is not None:
                total_loss += sent_sim_loss
            else:
                total_loss = sent_sim_loss
            if self.sentence_regularization:
                total_loss += sent_std_loss + (0.1 * sent_cov_loss)

        doc_sim_loss = None
        doc_std_loss = None
        doc_cov_loss = None
        pre_doc_std_loss = None
        pre_doc_cov_loss = None
        if self.config.doc_sim:
            # document projections similarity
            doc_sim_loss = 1 - self.cosine(primary_pooled_outputs, secondary_pooled_outputs).mean()
            # document projections variance, covariance
            doc_std_loss, doc_cov_loss = vic_reg(primary_pooled_outputs, secondary_pooled_outputs)
            total_loss += doc_sim_loss
            if self.document_regularization:
                total_loss += doc_std_loss + (0.1 * doc_cov_loss)

        if not return_dict:
            output = (primary_prediction_scores,) + primary_outputs[2:]
            return ((total_loss, masked_lm_loss, sent_sim_loss, doc_sim_loss) + output) if total_loss is not None else output

        return HATForVICRegPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            sent_sim_loss=sent_sim_loss,
            sent_std_loss=sent_std_loss,
            sent_cov_loss=sent_cov_loss,
            pre_sent_std_loss=pre_sent_std_loss,
            pre_sent_cov_loss=pre_sent_cov_loss,
            doc_sim_loss=doc_sim_loss,
            doc_std_loss=doc_std_loss,
            doc_cov_loss=doc_cov_loss,
            pre_doc_std_loss=pre_doc_std_loss,
            pre_doc_cov_loss=pre_doc_cov_loss,
            prediction_logits=primary_prediction_scores,
            hidden_states=primary_outputs.hidden_states,
            attentions=primary_outputs.attentions,
        )


@add_start_docstrings(
    """
    HAT Model with three heads on top as done during the pretraining: a `masked language modeling` head and a `document
    representation prediction ` head and a `masked sentence representation prediction ` head.
    """,
    HAT_START_DOCSTRING,
)
class HATModelForSimCLRPreTraining(HATPreTrainedModel):
    def __init__(self, config,
                 document_regularization=True,
                 sentence_regularization=True):
        super().__init__(config)

        self.document_regularization = document_regularization
        self.sentence_regularization = sentence_regularization
        self.hi_transformer = HATModel(config)
        if self.config.mlm:
            self.lm_head = HATLMHead(config)
        if self.config.sent_sim or self.config.doc_sim:
            self.sentencizer = HATSentencizer(config)
        if self.config.doc_sim:
            self.pooler = HATPooler(config, pooling='max')
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        secondary_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        secondary_labels=None,
        sentence_masks=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        primary_outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        secondary_outputs = self.hi_transformer(
            secondary_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Collect sequence output representations
        primary_sequence_output = primary_outputs[0]
        secondary_sequence_output = secondary_outputs[0]

        # Masked Language Modeling (MLM)
        primary_prediction_scores = None
        secondary_prediction_scores = None
        if self.config.mlm:
            primary_prediction_scores = self.lm_head(primary_sequence_output)
            if secondary_labels is not None:
                secondary_prediction_scores = self.lm_head(secondary_sequence_output)

        if self.config.sent_sim or self.config.doc_sim:
            primary_sentence_outputs = self.sentencizer(primary_sequence_output)
            secondary_sentence_outputs = self.sentencizer(secondary_sequence_output)

        # Document Representation Prediction (DRP)
        if self.config.doc_sim:
            primary_pooled_outputs = self.pooler(primary_sentence_outputs)
            secondary_pooled_outputs = self.pooler(secondary_sentence_outputs)

        total_loss = None
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(primary_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss.clone() / 2
        if secondary_labels is not None:
            masked_lm_loss = loss_fct(secondary_prediction_scores.view(-1, self.config.vocab_size), secondary_labels.view(-1))
            total_loss += masked_lm_loss / 2

        sent_contr_loss = None
        sent_std_loss = None
        sent_cov_loss = None
        if self.config.sent_sim:
            # sentence contrastive loss
            loss_fct = CrossEntropyLoss()
            # sentence queue: (2 x BS X S, H)
            flatten_sentence_masks = sentence_masks.view(-1)
            flatten_primary_sentence_outputs = primary_sentence_outputs.view(-1, self.config.hidden_size)
            flatten_secondary_sentence_outputs = secondary_sentence_outputs.view(-1, self.config.hidden_size)
            # merge sentence queue (sentences from both branches)
            flatten_primary_sentence_outputs = normalize(flatten_primary_sentence_outputs)
            flatten_secondary_sentence_outputs = normalize(flatten_secondary_sentence_outputs)
            sentence_queue = torch.cat([flatten_primary_sentence_outputs, flatten_secondary_sentence_outputs], dim=0)

            # sentence logits: (BS x S, 2 x BS x S)
            primary_sent_contrast_logits = torch.matmul(flatten_primary_sentence_outputs, sentence_queue.T) / self.config.temperature
            secondary_sent_contrast_logits = torch.matmul(flatten_secondary_sentence_outputs, sentence_queue.T) / self.config.temperature

            batch_size = primary_sent_contrast_logits.shape[0]

            # mask-out self-contrast cases
            logits_mask = torch.eye(batch_size, batch_size).to(input_ids.device)
            primary_logits_mask = torch.cat([logits_mask, torch.zeros_like(logits_mask).to(input_ids.device)], dim=1).to(input_ids.device)
            secondary_logits_mask = torch.cat([torch.zeros_like(logits_mask).to(input_ids.device), logits_mask], dim=1).to(input_ids.device)

            primary_sent_contrast_logits += (primary_logits_mask * -1e3)
            secondary_sent_contrast_logits += (secondary_logits_mask * -1e3)

            # mask-out logits in padded sentences
            primary_sent_contrast_logits[:, ~flatten_sentence_masks.repeat(2)] = -1e3
            primary_sent_contrast_logits[:, ~flatten_sentence_masks.repeat(2)] = -1e3

            # auto-compute labels
            primary_sentence_labels = torch.arange(batch_size).to(input_ids.device) + batch_size
            primary_sentence_labels[~flatten_sentence_masks] = -100
            secondary_sentence_labels = torch.arange(batch_size).to(input_ids.device)
            secondary_sentence_labels[~flatten_sentence_masks] = -100

            # compute loss for both branches
            sent_contr_loss = (loss_fct(primary_sent_contrast_logits, primary_sentence_labels) +
                                   loss_fct(secondary_sent_contrast_logits, secondary_sentence_labels)) * 0.5

            # sentence outputs variance, covariance
            sent_std_loss, sent_cov_loss = vic_reg(
                primary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size),
                secondary_sentence_outputs[sentence_masks].view(-1, self.config.hidden_size))
            if labels is not None:
                total_loss += sent_contr_loss
            else:
                total_loss = sent_contr_loss
            if self.sentence_regularization:
                total_loss += sent_std_loss + (0.1 * sent_cov_loss)

        doc_contr_loss = None
        doc_std_loss = None
        doc_cov_loss = None
        if self.config.doc_sim:
            # sentence contrastive loss
            loss_fct = CrossEntropyLoss()
            # sentence queue: (2 x BS, H)
            primary_pooled_outputs = normalize(primary_pooled_outputs)
            secondary_pooled_outputs = normalize(secondary_pooled_outputs)
            document_queue = torch.cat([primary_pooled_outputs, secondary_pooled_outputs], dim=0)

            # sentence logits: (BS, 2 x BS)
            primary_doc_contrast_logits = torch.matmul(primary_pooled_outputs, document_queue.T) / self.config.temperature
            secondary_doc_contrast_logits = torch.matmul(secondary_pooled_outputs, document_queue.T) / self.config.temperature

            batch_size = primary_doc_contrast_logits.shape[0]

            # mask-out self-contrast cases
            logits_mask = torch.eye(batch_size, batch_size).to(input_ids.device)
            primary_logits_mask = torch.cat([logits_mask, torch.zeros_like(logits_mask).to(input_ids.device)], dim=1).to(input_ids.device)
            secondary_logits_mask = torch.cat([torch.zeros_like(logits_mask).to(input_ids.device), logits_mask], dim=1).to(input_ids.device)

            primary_doc_contrast_logits += (primary_logits_mask * -1e3)
            secondary_doc_contrast_logits += (secondary_logits_mask * -1e3)

            # auto-compute labels
            primary_doc_labels = torch.arange(batch_size).to(input_ids.device) + batch_size
            secondary_doc_labels = torch.arange(batch_size).to(input_ids.device)

            # compute loss for both branches
            doc_contr_loss = (loss_fct(primary_doc_contrast_logits, primary_doc_labels) +
                              loss_fct(secondary_doc_contrast_logits, secondary_doc_labels)) * 0.5

            # sentence outputs variance, covariance
            doc_std_loss, doc_cov_loss = vic_reg(primary_pooled_outputs, secondary_pooled_outputs)
            if labels is not None:
                total_loss += doc_contr_loss
            else:
                total_loss = doc_contr_loss
            if self.document_regularization:
                total_loss += doc_std_loss + (0.1 * doc_cov_loss)

        if not return_dict:
            output = (primary_prediction_scores,) + primary_outputs[2:]
            return ((total_loss, masked_lm_loss, sent_contr_loss, doc_contr_loss) + output) if total_loss is not None else output

        return HATForSimCLRPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            sent_contr_loss=sent_contr_loss,
            sent_std_loss=sent_std_loss,
            sent_cov_loss=sent_cov_loss,
            doc_contr_loss=doc_contr_loss,
            doc_std_loss=doc_std_loss,
            doc_cov_loss=doc_cov_loss,
            prediction_logits=primary_prediction_scores,
            hidden_states=primary_outputs.hidden_states,
            attentions=primary_outputs.attentions,
        )


@add_start_docstrings(
    """
    HAT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    HAT_START_DOCSTRING,
)
class HATForSequenceClassification(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling='max'):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.max_sentence_length = config.max_sentence_length
        self.pooling = pooling

        self.hi_transformer = HATModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.pooler = HATPooler(config, pooling=pooling)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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
            pooled_output = self.pooler(sequence_output[:, ::self.max_sentence_length])

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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
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


@add_start_docstrings(""" HAT Model transformer for masked sentence representation prediction """,
    HAT_START_DOCSTRING,
)
class HATModelForSequentialSentenceClassification(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.hi_transformer = HATModel(config)
        self.sentencizer = HATSentencizer(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
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


@add_start_docstrings(
    """
    HAT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    HAT_START_DOCSTRING,
)
class HATForMultipleChoice(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooling='last'):
        super().__init__(config)

        self.pooling = pooling
        self.max_sentence_length = config.max_sentence_length
        self.hi_transformer = HATModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.pooler = HATPooler(config, pooling=pooling)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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

        outputs = self.hi_transformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
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
            pooled_output = self.pooler(sequence_output[:, ::self.max_sentence_length])

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


@add_start_docstrings(
    """
    HAT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    HAT_START_DOCSTRING,
)
class HATForTokenClassification(HATPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hi_transformer = HATModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    HAT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    HAT_START_DOCSTRING,
)
class HATForQuestionAnswering(HATPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.hi_transformer = HATModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HAT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    return position_ids[:, :input_ids.size(1)].repeat(input_ids.size(0), 1) * mask


def normalized_output_std_loss(x):
    return torch.std(x / torch.nn.functional.normalize(x, dim=1), dim=0).mean()


def vic_reg(x: torch.Tensor, y: torch.Tensor):
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.relu(1 - std_x)) / 2 + torch.mean(torch.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.shape[-1]) + \
               off_diagonal(cov_y).pow_(2).sum().div(y.shape[-1])

    return std_loss, cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

