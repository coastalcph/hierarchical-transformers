from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerTokenizer, HiTransformerConfig

DUMMY_TEXTS = ['dog ' * 8192, 'cat ' * 7000, 'mouse ' * 5000]

config = HiTransformerConfig.from_pretrained('../../data/hi-transformer')
tokenizer = HiTransformerTokenizer.from_pretrained('../../data/hi-transformer')
model = HiTransformerForMaskedLM.from_config(config)
pytorch_total_params = sum(p.numel() for p in model.hi_transformer.parameters())
batch = tokenizer(DUMMY_TEXTS, padding=True, truncation=True, return_tensors='pt')
outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
print()
