from models.hi_transformer_v2 import HiTransformerV2ForMaskedLM, HiTransformerV2Tokenizer, HiTransformerV2Config

DUMMY_TEXTS = [' '.join(['dog'] * 8192),  ' '.join(['cat'] * 7000), ' '.join(['mouse'] * 5000)]

config = HiTransformerV2Config.from_pretrained('../data/hi-transformer-v2')
tokenizer = HiTransformerV2Tokenizer.from_pretrained('../data/hi-transformer-v2')
model = HiTransformerV2ForMaskedLM.from_config(config)
pytorch_total_params = sum(p.numel() for p in model.hi_transformer.parameters())
batch = tokenizer(DUMMY_TEXTS, padding=True, truncation=True, return_tensors='pt')
outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
print()