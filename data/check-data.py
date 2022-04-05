from datasets import load_dataset

dataset = load_dataset('./wikipedia-datasets', '20200501.en')

print(f'Train subset: {len(dataset["train"])}')
print(f'Validation subset: {len(dataset["validation"])}')
print(f'Test subset: {len(dataset["test"])}')
