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
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch, _numpy_collate_batch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
@dataclass
class DataCollatorForPreTraining(DataCollatorMixin):
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
    tfidf_vect: TfidfVectorizer
    pca_solver: PCA
    mlm: bool = True
    mslm: bool = True
    drp: bool = True
    srp: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_sentence_length: int = 64

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
        original_inputs = copy.deepcopy(batch["input_ids"])
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        if self.mslm:
            batch["input_ids"], batch["labels"] = self.torch_mask_sentence_tokens(batch["input_ids"], batch["attention_mask"])
        if self.drp:
            batch["document_labels"] = self.torch_doc_reprs(original_inputs)
        if self.srp:
            batch["sentence_labels"] = self.torch_sentence_reprs(original_inputs)
        return batch

    def torch_doc_reprs(self, inputs: Any) -> Any:
        import torch
        tf_idfs = self.tfidf_vect.transform(self.tokenizer.batch_decode(inputs))
        features = self.pca_solver.transform(tf_idfs.toarray())
        doc_reps = torch.Tensor(features)

        return doc_reps

    def torch_sentence_reprs(self, inputs: Any) -> Any:
        import torch
        sent_reprs = []
        for sample_ids in inputs:
            tf_idfs = self.tfidf_vect.transform(self.tokenizer.batch_decode([sample_ids[i:i + self.max_sentence_length] for i in range(0, len(sample_ids), self.max_sentence_length)]))
            features = self.pca_solver.transform(tf_idfs.toarray())
            sent_reprs.append(torch.Tensor(features))

        return torch.stack(sent_reprs)

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

    def torch_mask_sentence_tokens(self, inputs: Any, attention_mask: Any) -> Tuple[Any, Any]:
        """
        Prepare masked sentence tokens inputs/labels for masked sentence language modeling.
        """
        import torch

        labels = inputs.clone()

        for idx, input in enumerate(inputs):
            # Find padded sentences
            sentence_mask = [attention_mask[idx][i:i + self.max_sentence_length].sum() != 0
                             for i in range(0, len(input), self.max_sentence_length)]
            sentence_mask = torch.tensor(sentence_mask, dtype=torch.bool)
            # We sample a few sentences in each sequence for MSLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(sentence_mask.shape, self.mlm_probability)
            probability_matrix.masked_fill_(~sentence_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            for sent_idx, mask in enumerate(masked_indices):
                if mask:
                    # Mask sentence tokens except <cls>
                    labels[idx][sent_idx * self.max_sentence_length] = -100
                    # Find non-padded sentence tokens
                    no_padded_ids = (inputs[idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length] != self.tokenizer.pad_token_id).bool()
                    # Mask non-padded sentence tokens
                    inputs[idx][sent_idx * self.max_sentence_length + 1:(sent_idx + 1) * self.max_sentence_length][no_padded_ids] = self.tokenizer.mask_token_id
                else:
                    # We only compute loss on masked sentence tokens
                    labels[idx][sent_idx * self.max_sentence_length:(sent_idx + 1) * self.max_sentence_length] \
                        = -100
            # Do not compute loss on padded sequence tokens
            labels[idx][~attention_mask[idx].bool()] = -100

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

