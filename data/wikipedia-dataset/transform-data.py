import json
import random

import tqdm
from datasets import load_dataset


def create_streamable_wiki():
    dataset = load_dataset('wikipedia', '20200501.en')['train']

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


def create_eval_streamable_wiki():
    for subset, size in zip(['train', 'validation', 'test'], [60000, 10000, 10000]):
        dataset = load_dataset('./data/wikipedia-dataset', '20200501.en', data_dir='./data/wikipedia-dataset', split=subset, streaming=True)
        valid_samples = []
        for sample in tqdm.tqdm(iter(dataset)):
            if 1000 > len(sample['text'].split()) > 500:
                valid_samples.append(sample['text'])

        print(f'TOTAL SAMPLES MEET CRITERIA: {len(valid_samples)}')
        samples = random.sample(valid_samples, k=size)

        with open(f'evaluation_{subset}.jsonl', 'w') as file:
            for sample in samples:
                file.write(json.dumps({'text': sample})+'\n')


if __name__ == '__main__':
    create_eval_streamable_wiki()
