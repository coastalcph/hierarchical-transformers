from datasets import load_dataset

dataset = load_dataset('data/wikipedia-dataset', '20200501.en', data_dir='data/wikipedia-dataset')

print(f'Train subset: {len(dataset["train"])}')
print(f'Validation subset: {len(dataset["validation"])}')
print(f'Test subset: {len(dataset["test"])}')
