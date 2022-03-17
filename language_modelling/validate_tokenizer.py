from transformers import AutoTokenizer, RobertaConfig
from datasets import load_dataset

ROBERTA_TOK_FOLDER = '../data/custom-tokenizer'

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
custom_roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_TOK_FOLDER, config=RobertaConfig.from_pretrained('roberta-base'))
roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
for example in dataset['text'][:100]:
    print(example)
    print('-' * 200)
    input_ids = custom_roberta_tokenizer.encode(example)
    print(custom_roberta_tokenizer.convert_ids_to_tokens(input_ids))
    print('-' * 200)
    input_ids = roberta_tokenizer.encode(example)
    print(roberta_tokenizer.convert_ids_to_tokens(input_ids))
    print('-' * 200)
    print('-' * 200)
