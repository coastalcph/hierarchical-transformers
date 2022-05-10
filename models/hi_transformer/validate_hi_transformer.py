from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerTokenizer, HiTransformerConfig
from datasets import load_dataset

DUMMY_TEXTS = load_dataset('multi_eurlex', 'en', split='test')['text'][:64]
MODEL_PATH = '../../data/PLMs/hi-transformer-p1-grouped'

config = HiTransformerConfig.from_pretrained(MODEL_PATH)
tokenizer = HiTransformerTokenizer.from_pretrained(MODEL_PATH)
model = HiTransformerForMaskedLM.from_pretrained(MODEL_PATH, config=config)
pytorch_total_params = sum(p.numel() for p in model.hi_transformer.parameters())
batch = tokenizer(DUMMY_TEXTS, padding=True, truncation=True, return_tensors='pt', greedy_chunking=False)
outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
