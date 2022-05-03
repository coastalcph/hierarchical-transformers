"""Tokenization classes for Hi-Transformer."""
import torch
from transformers import AutoTokenizer
from transformers.models.longformer.configuration_longformer import LongformerConfig
from transformers.utils import logging
from nltk import sent_tokenize
logger = logging.get_logger(__name__)


class LongformerTokenizer:
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.config = LongformerConfig.from_pretrained(self._tokenizer.name_or_path)
        # hardcoded values
        self.config.max_sentence_size = 128
        self.config.max_sentence_length = 128
        self.config.max_sentences = 8
        self._tokenizer.model_max_length = self.model_max_length
        self.type2id = {'input_ids': (self._tokenizer.sep_token_id, self._tokenizer.pad_token_id),
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

        for idx, _ in enumerate(input_ids):
            input_ids[idx][0] = self._tokenizer.cls_token_id

        return input_ids

    def get_special_tokens_mask(self, *args, **kwargs):
        return self._tokenizer.get_special_tokens_mask(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs))

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained( *args, **kwargs)

    def __call__(self, texts, **kwargs):
        greedy_chunking = kwargs.pop('greedy_chunking', None)
        if greedy_chunking:
            # fixed uniform chunking
            batch = self.uniform_chunking(texts, **kwargs)
        else:
            # dynamic sentence splitting and grouping
            batch = self.sentence_splitting(texts, **kwargs)

        for idx, _ in enumerate(batch['input_ids']):
            batch['input_ids'][idx][0] = self._tokenizer.cls_token_id

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


if __name__ == "__main__":
    tokenizer = LongformerTokenizer.from_pretrained('roberta-base')
    inputs = tokenizer([' '.join(['dog'] * 8192),
                        ' '.join(['cat'] * 7000),
                       ' '.join(['mouse'] * 5000)],
                       padding=True, max_length=8192, truncation=True
                       )
    print()
