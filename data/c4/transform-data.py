import json
import random

import tqdm
from datasets import load_dataset


def create_eval_streamable_wiki():
    for subset, size in zip(['train', 'validation'], [60000, 20000]):
        dataset = load_dataset('c4', 'en', split=subset, streaming=True)
        valid_samples = []
        for sample in tqdm.tqdm(iter(dataset)):
            if 4000 > len(sample['text'].split()) > 1000:
                valid_samples.append(sample['text'])

        print(f'TOTAL SAMPLES MEET CRITERIA: {len(valid_samples)}')
        samples = random.sample(valid_samples, k=size)

        with open(f'evaluation_{subset}.jsonl', 'w') as file:
            for sample in samples:
                file.write(json.dumps({'text': sample})+'\n')


if __name__ == '__main__':
    create_eval_streamable_wiki()
