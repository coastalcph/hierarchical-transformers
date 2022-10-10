from models.hat import HATForMaskedLM, HATTokenizer, HATConfig
from datasets import load_dataset

DUMMY_TEXTS = load_dataset('multi_eurlex', 'en', split='test')['text'][:64]
MODEL_PATH = '../../data/PLMs/hi-transformer-p1-grouped'

config = HATConfig.from_pretrained(MODEL_PATH)
tokenizer = HATTokenizer.from_pretrained(MODEL_PATH)
model = HATForMaskedLM.from_pretrained(MODEL_PATH, config=config)
pytorch_total_params = sum(p.numel() for p in model.hat.parameters())
batch = tokenizer(DUMMY_TEXTS, padding=True, truncation=True, return_tensors='pt', greedy_chunking=False)
outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
