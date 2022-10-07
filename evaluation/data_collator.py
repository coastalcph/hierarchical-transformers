from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorForMultiLabelClassification:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        import torch
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["labels"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch
