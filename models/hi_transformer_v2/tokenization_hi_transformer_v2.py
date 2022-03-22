"""Tokenization classes for Hi-Transformer."""
import torch
from transformers import AutoTokenizer
from .configuration_hi_transformer_v2 import HiTransformerV2Config
from transformers.utils import logging
logger = logging.get_logger(__name__)


class HiTransformerV2Tokenizer:
    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer
        self.config = HiTransformerV2Config.from_pretrained(self._tokenizer.name_or_path)
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

    def encode(self, text, **kwargs):
        input_ids = self._tokenizer.encode_plus(text, add_special_tokens=False, **kwargs)
        input_ids = self.chunks(input_ids[: self.model_max_length - self.config.max_sentences],
                                chunk_size=self.config.max_sentence_length, special_id=self.type2id['input_ids'])
        return input_ids

    def get_special_tokens_mask(self, *args, **kwargs):
        return self._tokenizer.get_special_tokens_mask(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return cls(tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs))

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained( *args, **kwargs)

    def __call__(self, texts, **kwargs):
        batch = self._tokenizer(texts, add_special_tokens=False,  **kwargs)
        batch_out = {input_type: [] for input_type in batch}
        for input_type in batch:
            fixed_batch = []
            for example in batch[input_type]:
                fixed_batch.append(self.chunks(example[: self.model_max_length - self.config.max_sentences],
                                               chunk_size=self.config.max_sentence_length,
                                               special_id=self.type2id[input_type]))
            batch_out[input_type] = fixed_batch if isinstance(fixed_batch[0], list) else torch.stack(fixed_batch)
        return batch_out

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


if __name__ == "__main__":
    tokenizer = HiTransformerV2Tokenizer.from_pretrained('roberta-base')
    inputs = tokenizer([' '.join(['dog'] * 8192),
                        ' '.join(['cat'] * 7000),
                       ' '.join(['mouse'] * 5000)],
                       padding=True, max_length=8192, truncation=True
                       )
    print()
