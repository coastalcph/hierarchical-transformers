from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

config = AutoConfig.from_pretrained('../../data/roberta')
tokenizer = AutoTokenizer.from_pretrained('../../data/roberta')
model = AutoModelForMaskedLM.from_config(config)
