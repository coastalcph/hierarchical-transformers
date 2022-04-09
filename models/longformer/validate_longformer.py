from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from models.longformer import LongformerForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('../../data/PLMs/longformer')
model = LongformerForMaskedLM.from_pretrained('../../data/PLMs/longformer')
batch = tokenizer(['this is a dog', 'this is a cat'], return_tensors='pt')
model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])