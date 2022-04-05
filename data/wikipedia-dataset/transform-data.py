import json
import tqdm
from datasets import load_dataset

dataset = load_dataset('wikipedia', '20200501.en')['train']

dataset_size = len(dataset['title'])

file = open('train.jsonl', 'w')
for idx, (doc_title, doc_text) in tqdm.tqdm(enumerate(zip(dataset['title'], dataset['text']))):
    if idx == 5500000:
        file.close()
        file = open('dev.jsonl', 'w')
    elif idx == 6000000:
        file.close()
        file = open('test.jsonl', 'w')

    file.write(json.dumps({'title': doc_title, 'text': doc_text}) + '\n')

file.close()