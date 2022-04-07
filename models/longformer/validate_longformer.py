from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

config = AutoConfig.from_pretrained('../../data/longformer')
tokenizer = AutoTokenizer.from_pretrained('../../data/longformer')
model = AutoModelForMaskedLM.from_config(config)
