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
"""Tokenization classes for HAT."""
import torch
from transformers import RobertaTokenizer, BertTokenizer
from .configuration_hat import HATConfig
from transformers.utils import logging
try:
    from nltk import sent_tokenize
except:
    raise Exception('NLTK is not installed! Install it with `pip install nltk`...')
logger = logging.get_logger(__name__)


class HATTokenizer:
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.config = HATConfig.from_pretrained(self._tokenizer.name_or_path)
        self._tokenizer.model_max_length = self.model_max_length
        self.type2id = {'input_ids': (self._tokenizer.cls_token_id, self._tokenizer.pad_token_id),
                        'token_type_ids': (0, 0),
                        'attention_mask': (1, 0),
                        'special_tokens_mask': (1, -100)}

    @property
    def model_max_length(self):
        return self.config.model_max_length

    @property
    def mask_token(self):
        return self._tokenizer.mask_token

    @property
    def mask_token_id(self):
        return self._tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self._tokenizer.sep_token_id

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        return len(self._tokenizer)

    def pad(self, *args, **kwargs):
        return self._tokenizer.pad(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self._tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self._tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        return self._tokenizer.tokenize(text, **kwargs)

    def encode(self, text, **kwargs):
        input_ids = self._tokenizer.encode_plus(text, add_special_tokens=False, **kwargs)
        input_ids = self.chunks(input_ids[: self.model_max_length - self.config.max_sentences],
                                chunk_size=self.config.max_sentence_length, special_id=self.type2id['input_ids'])
        return input_ids

    def get_special_tokens_mask(self, *args, **kwargs):
        return self._tokenizer.get_special_tokens_mask(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except:
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(tokenizer=tokenizer)

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained( *args, **kwargs)

    def __call__(self, text, **kwargs):
        greedy_chunking = kwargs.pop('greedy_chunking', None)
        text_pair = kwargs.pop('text_pair', None)
        if isinstance(text[0], list):
            batch = self.auto_chunking(text, **kwargs)
        elif greedy_chunking:
            # fixed uniform chunking
            batch = self.uniform_chunking(text, **kwargs)
        else:
            # dynamic sentence splitting and grouping
            batch = self.sentence_splitting(text, **kwargs)

        if text_pair:
            batch_b = self._tokenizer(text_pair, add_special_tokens=False,
                                      padding=False, truncation=False)
            for idx, sample in enumerate(batch['input_ids']):
                n_sentences = sum(sample[::self.config.max_sentence_size])
                for input_key in batch:
                    batch[input_key][idx][self.config.max_sentence_size * n_sentences:
                                          self.config.max_sentence_size * (n_sentences + 1)] = \
                        self.pad_sentence(batch_b[input_key][idx],
                                          special_id=(self.sep_token_id, self.pad_token_id)
                                          if input_key == 'input_ids' else self.type2id[input_key])

        return batch

    def uniform_chunking(self, texts, **kwargs):
        original_batch = self._tokenizer(texts, add_special_tokens=False, **kwargs)
        batch = {input_type: [] for input_type in original_batch}
        for input_type in original_batch:
            fixed_batch = []
            for example in original_batch[input_type]:
                fixed_batch.append(self.chunks(example[: self.model_max_length - self.config.max_sentences],
                                               chunk_size=self.config.max_sentence_length,
                                               special_id=self.type2id[input_type]))
            batch[input_type] = fixed_batch if isinstance(fixed_batch[0], list) else torch.stack(fixed_batch)

        if kwargs['padding']:
            batch = self.pad(batch,
                             padding=kwargs['padding'],
                             max_length=kwargs['max_length'],
                             pad_to_multiple_of=kwargs['max_length'])

        return batch

    def auto_chunking(self, texts, **kwargs):
        batch = {}
        for text_idx, text in enumerate(texts):
            example_batch = self._tokenizer(text, add_special_tokens=False, **kwargs)
            for input_key in example_batch:
                key_inputs_list = []
                for idx, example in enumerate(example_batch[input_key][:self.config.max_sentences]):
                    key_inputs_list.append(self.pad_sentence(example, special_id=self.type2id[input_key]))
                if isinstance(key_inputs_list[0], list):
                    key_inputs_list = [token for sentence in key_inputs_list for token in sentence]
                else:
                    key_inputs_list = torch.stack([token for sentence in key_inputs_list for token in sentence])
                if input_key in batch:
                    batch[input_key].append(key_inputs_list)
                else:
                    batch[input_key] = [key_inputs_list]

        if kwargs['padding']:
            batch = self.pad(batch,
                             padding=kwargs['padding'],
                             max_length=kwargs['max_length'],
                             pad_to_multiple_of=kwargs['max_length'])

        return batch

    def chunks(self, flat_inputs, chunk_size=128, special_id=0):
        if isinstance(flat_inputs, list):
            return self.list_chunks(flat_inputs, chunk_size, special_id)
        else:
            return self.tensor_chunks(flat_inputs, chunk_size, special_id)

    def list_chunks(self, flat_inputs, chunk_size=128, special_id=(0, 0)):
        """Yield successive n-sized chunks from lst."""
        structured_inputs = [[special_id[0] if sum(flat_inputs[i:i + chunk_size-1]) else special_id[1]]
                             + flat_inputs[i:i + chunk_size-1] for i in range(0, len(flat_inputs), chunk_size-1)]
        return [token_input for sentence_inputs in structured_inputs for token_input in sentence_inputs]

    def tensor_chunks(self, flat_inputs, chunk_size=128, special_id=(0, 0)):
        """Yield successive n-sized chunks from lst."""
        structured_inputs = torch.stack([torch.cat((torch.tensor([special_id[0] if flat_inputs[i:i + chunk_size-1].sum() else special_id[1]], dtype=torch.int),
                                                    flat_inputs[i:i + chunk_size-1])) for i in range(0, len(flat_inputs), chunk_size-1)])
        return structured_inputs.reshape(-1)

    def sentence_splitting(self, texts, **kwargs):
        fixed_batch = []
        doc_out = {}
        for text in texts:
            # sentence splitting
            sentences = sent_tokenize(text)
            # tokenization of sentences
            sentences = self._tokenizer(sentences, add_special_tokens=False, padding=False, truncation=False)
            # sentence grouping - merging short sentences to minimize padding
            doc_out = self.sentence_grouping(sentences)
            fixed_batch.append(doc_out)
        # batchify examples
        batch = {input_type: [] for input_type in doc_out}
        for input_type in batch:
            batch[input_type] = [example[input_type] for example in fixed_batch]
            if not isinstance(batch[input_type][0], list):
                batch[input_type] = torch.stack(batch[input_type])

        if kwargs['padding']:
            batch = self.pad(batch,
                             padding=kwargs['padding'],
                             max_length=kwargs['max_length'],
                             pad_to_multiple_of=kwargs['max_length'])

        return batch

    def sentence_grouping(self, sentences):
        doc_out = {input_type: [] for input_type in sentences}
        for input_type in sentences:
            tmp_doc = []
            tmp_sentence = []
            for example in sentences[input_type]:
                if len(tmp_doc) >= self.config.max_sentences:
                    break
                if len(tmp_sentence) + len(example) <= self.config.max_sentence_length - 1:
                    tmp_sentence.extend(example)
                else:
                    tmp_doc.append(self.pad_sentence(tmp_sentence if len(tmp_sentence) else example,
                                                     chunk_size=self.config.max_sentence_length,
                                                     special_id=self.type2id[input_type]))
                    tmp_sentence = example if len(tmp_sentence) else example[self.config.max_sentence_length:]
            if len(tmp_sentence) and len(tmp_doc) < self.config.max_sentences:
                tmp_doc.append(self.pad_sentence(tmp_sentence,
                                                 chunk_size=self.config.max_sentence_length,
                                                 special_id=self.type2id[input_type]))
            doc_out[input_type] = [token for sentence in tmp_doc for token in sentence]
        return doc_out

    def pad_sentence(self, flat_input, chunk_size=128, special_id=(0, 0)):
        if isinstance(flat_input, list):
            return [special_id[0]] + flat_input[:chunk_size-1] + [self.pad_token_id] * max(0, chunk_size - len(flat_input) - 1)
        else:
            return torch.cat((torch.tensor([special_id[0] if flat_input[:chunk_size-1].sum()
                                            else special_id[1]], dtype=torch.int),
                              flat_input[:chunk_size-1],
                              torch.tensor([self.pad_token_id] * max(0, chunk_size - len(flat_input) - 1), dtype=torch.int)
                              ))
        
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.
        <Tip warning={true}>
        This API is experimental and may have some slight breaking changes in the next releases.
        </Tip>
        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"TFAutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

