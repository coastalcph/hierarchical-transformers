# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import random
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import chain

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch, _numpy_collate_batch
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForBoWPreTraining(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        tfidf_vect ('TFIDFVectorizer',  *optional*, defaults to `None`):
            TFIDFVectorizer
        pca_solver ('sklearn.PCA',  *optional*, defaults to `None`):
            PCA Solver
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mslm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked sentence language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token in full sentences.
        drp (`bool`, *optional*, defaults to `True`):
            Whether or not to use document prediction. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        srp (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked sentence prediction. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        sentence_embedder (`SentenceTransformer`, *optional*, defaults to `None`):
            Sentence BERT model to encode masked sentences.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mslm: bool = True
    drp: bool = True
    srp: bool = True
    sentence_embedder: Optional[bool] = None
    sentence_embedding_size: Optional[int] = 256
    mlm_probability: float = 0.15
    ms_probability: float = 0.25
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_sentence_length: int = 64
    bow_unused_mask: List[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # Clone original input ids to consider in non-mlm tasks
        original_input_ids = batch['input_ids'].clone()
        if self.mslm or self.srp or self.drp:
            batch['sentence_masks'] = self.torch_mask_sentences(batch['attention_mask'])
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], special_tokens_mask=special_tokens_mask)
        if self.mslm:
            batch["input_ids"], batch["labels"] = self.torch_mask_sentence_tokens(batch["input_ids"], batch["attention_mask"], batch['sentence_masks'])
        if self.srp and self.sentence_embedder is not None:
            batch["input_ids"], batch["labels"], batch["sentence_labels"] = self.torch_sentence_reprs(batch["input_ids"], batch["labels"], original_input_ids, batch['sentence_masks'])
        elif self.srp:
            batch["input_ids"], batch["labels"], batch["sentence_labels"], batch['sentence_mask_ids'] = \
                self.torch_bow_sentence_labels(batch["input_ids"], batch["labels"], original_input_ids, batch['sentence_masks'])
        if self.drp:
            batch["document_labels"], batch['document_mask_ids'] = \
                self.torch_bow_document_labels(original_input_ids)

        return batch

    def torch_bow_document_labels(self, original_input_ids: Any) -> Any:
        import torch
        # sample random word indices
        probability_matrix = torch.full((len(self.tokenizer),), 0.01)
        # mask off unused words
        if self.bow_unused_mask is not None:
            probability_matrix.masked_fill_(self.bow_unused_mask, value=0.0)
        sample_ids = torch.bernoulli(probability_matrix).bool()
        # build document labels
        document_labels = torch.zeros((original_input_ids.shape[0], len(self.tokenizer)), dtype=torch.float)
        for doc_idx, doc_original_input_ids in enumerate(original_input_ids):
            unique_ids = torch.unique(doc_original_input_ids)
            document_labels[doc_idx][unique_ids] = 1
            sample_ids[unique_ids] = True
        # re-mask off unused words
        if self.bow_unused_mask is not None:
            sample_ids[self.bow_unused_mask] = False

        return document_labels[:, sample_ids], sample_ids

    def torch_bow_sentence_labels(self, inputs: Any, labels: Any, original_inputs: Any, sentence_masks: Any) -> Any:
        import torch
        # sample random word indices
        probability_matrix = torch.full((len(self.tokenizer),), 0.01)
        # mask off unused words
        if self.bow_unused_mask is not None:
            probability_matrix.masked_fill_(self.bow_unused_mask, value=0.0)
        sample_ids = torch.bernoulli(probability_matrix).bool()
        forced_masked_id = random.choice([1, 2, 3, 4, 5, 6])
        # build sentence labels
        sentence_labels = torch.zeros((inputs.shape[0], sentence_masks.shape[1], len(self.tokenizer)), dtype=torch.float)
        for doc_idx, sentence_mask in enumerate(sentence_masks):
            for sent_idx, mask_id in enumerate(sentence_mask):
                if mask_id:
                    # compute sentence-level bow representations
                    unique_ids = torch.unique(original_inputs[doc_idx][(sent_idx * self.max_sentence_length) + 1: (sent_idx+1) * self.max_sentence_length])
                    sentence_labels[doc_idx][sent_idx][unique_ids] = 1
                    # include sentence ids in sampled ids
                    sample_ids[unique_ids] = True
                    if sentence_mask.sum() > 1:
                        # mask sentence tokens, except cls and pads
                        non_padded_ids = (inputs[doc_idx][(sent_idx * self.max_sentence_length) + 1:(sent_idx + 1) * self.max_sentence_length] != self.tokenizer.pad_token_id).bool()
                        inputs[doc_idx][(sent_idx * self.max_sentence_length) + 1: (sent_idx + 1) * self.max_sentence_length][non_padded_ids] = self.tokenizer.mask_token_id
                        # exclude sentence tokens from mlm loss
                        labels[doc_idx][(sent_idx * self.max_sentence_length) + 1: (sent_idx + 1) * self.max_sentence_length][non_padded_ids] = -100

            # mlm in a single word for safety
            inputs[doc_idx][forced_masked_id] = original_inputs[doc_idx][forced_masked_id]
            labels[doc_idx][forced_masked_id] = original_inputs[doc_idx][forced_masked_id]

        # re-mask off unused words
        if self.bow_unused_mask is not None:
            sample_ids[self.bow_unused_mask] = False

        return inputs, labels, sentence_labels[:, :, sample_ids], sample_ids

    def torch_doc_reprs(self, inputs: Any) -> Any:
        import torch
        tf_idfs = self.tfidf_vect.transform(self.tokenizer.batch_decode(inputs))
        features = self.pca_solver.transform(tf_idfs.toarray())
        doc_reps = torch.Tensor(features)

        return doc_reps

    def torch_sentence_reprs(self, inputs: Any, labels: Any, original_inputs: Any, sentence_masks: Any) -> Any:
        import torch
        sent_representations = torch.zeros((inputs.shape[0], sentence_masks.shape[1], self.sentence_embedding_size), dtype=torch.float)
        for doc_idx, sentence_mask in enumerate(sentence_masks):
            for sent_idx, mask_id in enumerate(sentence_mask):
                if mask_id:
                    masked_sentence = self.tokenizer.decode(original_inputs[doc_idx][(sent_idx * self.max_sentence_length): (sent_idx+1) * self.max_sentence_length], skip_special_tokens=True)
                    sent_representations[doc_idx][sent_idx] = self.sentence_embedder.encode(masked_sentence, convert_to_tensor=True, normalize_embeddings=True)
                    if sentence_mask.sum() > 1:
                        # mask sentence tokens, except cls and pads
                        non_padded_ids = (inputs[doc_idx][(sent_idx * self.max_sentence_length) + 1:(sent_idx + 1) * self.max_sentence_length] != self.tokenizer.pad_token_id).bool()
                        inputs[doc_idx][(sent_idx * self.max_sentence_length) + 1: (sent_idx + 1) * self.max_sentence_length][non_padded_ids] = self.tokenizer.mask_token_id
                        # exclude sentence tokens from mlm loss
                        labels[doc_idx][(sent_idx * self.max_sentence_length) + 1: (sent_idx + 1) * self.max_sentence_length][non_padded_ids] = -100

            # mlm in a few words for safety
            available_indices = torch.arange(0, inputs[doc_idx].shape[0]).numpy()
            forced_masked_id = random.choice(available_indices)
            inputs[doc_idx][forced_masked_id] = original_inputs[doc_idx][forced_masked_id]
            labels[doc_idx][forced_masked_id] = original_inputs[doc_idx][forced_masked_id]

        return inputs, labels, sent_representations

    def torch_mask_sentences(self, attention_mask: Any) -> Any:
        """
        Define masked sentences for masked sentence tasks.
        """
        import torch

        sentence_masks = torch.zeros((attention_mask.shape[0], int(len(attention_mask[0])/self.max_sentence_length)), dtype=torch.long)

        for idx, _ in enumerate(attention_mask):
            # Find padded sentences
            sentence_mask = [attention_mask[idx][i:i + self.max_sentence_length].sum() != self.tokenizer.pad_token_id
                             for i in range(0, len(attention_mask[0]), self.max_sentence_length)]
            sentence_mask = torch.tensor(sentence_mask, dtype=torch.bool)
            # We sample a few sentences in each sequence for MSLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(sentence_mask.shape, self.ms_probability)
            probability_matrix.masked_fill_(~sentence_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).int()
            if masked_indices.sum() == 0:
                masked_indices[0] = 1
            sentence_masks[idx] = masked_indices

        return sentence_masks

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_mask_sentence_tokens(self, inputs: Any, attention_masks: Any, sentence_masks: Any) -> Tuple[Any, Any]:
        """
        Prepare masked sentence tokens inputs/labels for masked sentence language modeling.
        """
        import torch

        labels = inputs.clone()

        for doc_idx, sentence_mask in enumerate(sentence_masks):
            for sent_idx, mask in enumerate(sentence_mask):
                if mask:
                    # We sample most sub-words in each sentence for MSLM training (with high probability 60%)
                    padded_ids = (inputs[doc_idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length] == self.tokenizer.pad_token_id).bool()
                    sent_probability_matrix = torch.full((self.max_sentence_length-1, ), self.mlm_probability)
                    sent_probability_matrix.masked_fill_(padded_ids, value=0.0)
                    token_masked_indices = torch.bernoulli(sent_probability_matrix).bool()
                    # Mask sentence tokens except <cls> and non masked tokens
                    labels[doc_idx][sent_idx * self.max_sentence_length] = -100
                    labels[doc_idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length][~token_masked_indices] = -100
                    inputs[doc_idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length][token_masked_indices] = self.tokenizer.mask_token_id
                else:
                    # We only compute loss on masked sentence tokens
                    labels[doc_idx][sent_idx * self.max_sentence_length:(sent_idx + 1) * self.max_sentence_length] = -100
            # Do not compute loss on padded sequence tokens
            labels[doc_idx][~attention_masks[doc_idx].bool()] = -100

        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import numpy as np

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import numpy as np

        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(np.bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(np.bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(np.bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForSiamesePreTraining(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        tfidf_vect ('TFIDFVectorizer',  *optional*, defaults to `None`):
            TFIDFVectorizer
        pca_solver ('sklearn.PCA',  *optional*, defaults to `None`):
            PCA Solver
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        similarity (`bool`, *optional*, defaults to `True`):
            Whether or not to use similarity pre-training objectives.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    similarity: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_sentence_length: int = 64
    sentence_masking: bool = False
    ms_probability: float = 0.25
    complementary_masking: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # Clone original input ids to consider in non-mlm tasks
        original_input_ids = batch['input_ids'].clone()
        secondary_original_input_ids = batch['input_ids'].clone()
        if self.mlm or self.similarity:
            batch["input_ids"], batch["labels"] = \
                self.torch_mask_tokens(original_input_ids, special_tokens_mask=special_tokens_mask,
                                       mlm_probability=self.mlm_probability, prior_masks=None)
        if self.similarity and self.sentence_masking:
            batch["secondary_input_ids"], batch['sentence_masks'] = \
                self.torch_mask_sentences(secondary_original_input_ids, batch['attention_mask'])
        else:
            batch['sentence_masks'] = torch.zeros((batch['input_ids'].shape[0],
                                                   batch['input_ids'].shape[1] // self.max_sentence_length),
                                                  dtype=torch.bool, device=batch['input_ids'].device)
            for idx, example in enumerate(secondary_original_input_ids):
                batch['sentence_masks'][idx] = (example[::self.max_sentence_length] != self.tokenizer.pad_token_id).bool()
            batch["secondary_input_ids"], batch["secondary_labels"] = \
                self.torch_mask_tokens(secondary_original_input_ids, special_tokens_mask=special_tokens_mask,
                                       mlm_probability=self.mlm_probability, prior_masks=(batch["labels"] != -100))

        if not self.mlm:
            batch.pop("labels", None)
            batch.pop("secondary_labels", None)

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, mlm_probability: Any = 0.2,
                          prior_masks: Any = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if self.complementary_masking and prior_masks is not None:
            probability_matrix.masked_fill_(prior_masks, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_mask_sentences(self, inputs: Any, attention_mask: Any) -> Any:
        """
        Define masked sentences for masked sentence tasks.
        """
        sentence_masks = torch.zeros((inputs.shape[0], inputs.shape[1] // self.max_sentence_length),
                                     dtype=torch.bool, device=inputs.device)

        for doc_idx, _ in enumerate(attention_mask):
            # Find padded sentences
            sentence_mask = (attention_mask[doc_idx][::self.max_sentence_length] != self.tokenizer.pad_token_id).bool()
            # We sample a few sentences in each sequence (with probability `self.ms_probability`)
            probability_matrix = torch.full(sentence_mask.shape, self.ms_probability)
            probability_matrix.masked_fill_(~sentence_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).int()
            if masked_indices.sum() == 0:
                masked_indices[0] = 1
            sentence_masks[doc_idx] = masked_indices.bool()
            for sent_idx, mask_id in enumerate(sentence_masks[doc_idx]):
                if mask_id:
                    inputs[doc_idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length] \
                        = self.tokenizer.mask_token_id

        return inputs, sentence_masks

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import numpy as np

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import numpy as np

        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(np.bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(np.bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(np.bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor([label[0] for label in labels], dtype=torch.int64)

        # Check-up for MCQA alternatives
        # for inputs, label in zip(batch['input_ids'], batch['labels']):
        #     print('-' * 100)
        #     print('-' * 100)
        #     print(f'Example: {self.tokenizer.decode(inputs[label.numpy()][:-128])}')
        #     print('-' * 100)
        #     for i in range(5):
        #         print(f'Answer [{i}]: {self.tokenizer.decode(inputs[i][-128:])}')
        #         print('-' * 100)
        #     print(f'Label: {label.numpy()}')
        #     print('-'*100)
        #     print('-' * 100)

        # Check-up for QUALITY alternatives
        # for inputs, label in zip(batch['input_ids'], batch['labels']):
        #     print('-' * 100)
        #     print('-' * 100)
        #     print(f'Example: {self.tokenizer.decode(inputs[0][:-256])}')
        #     print(f'Question: {self.tokenizer.decode(inputs[0][-256:-128])}')
        #     print('-' * 100)
        #     for i in range(4):
        #         print(f'Answer [{i}]: {self.tokenizer.decode(inputs[i][-128:])}')
        #         print('-' * 100)
        #     print(f'Label: {label.numpy()}')
        #     print('-'*100)
        #     print('-' * 100)

        return batch


@dataclass
class DataCollatorForSentenceOrder:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        import torch
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        # Check-up examples
        # for example_ids, labels in zip(batch['input_ids'], batch['labels']):
        #     for idx in range(8):
        #         print('-' * 100)
        #         print(f'{idx+1} [Label={int(labels[idx])}]:', self.tokenizer.decode(example_ids[idx*128:(idx+1)*128]))
        #     print('-' * 100)
        #     print('Labels:')
        #     print('-' * 100)

        return batch

@dataclass
class DataCollatorForDocumentClassification:
    tokenizer: PreTrainedTokenizerBase
    label_names: Dict

    def __call__(self, features):
        import torch
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        # Check-up examples
        # for example_ids, labels in zip(batch['input_ids'], batch['labels']):
        #     for idx in range(8):
        #         print('-' * 100)
        #         print(f'S{idx+1}:', self.tokenizer.decode(example_ids[idx*128:(idx+1)*128]))
        #     print('-' * 100)
        #     print(f'Labels: {[self.label_names[idx] for idx, label in enumerate(labels) if label==1]}')
        #     print('-' * 100)

        return batch

