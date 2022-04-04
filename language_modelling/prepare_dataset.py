import numpy as np
import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
import argparse


def main():
    """ set default hyperparams in default_hyperparams.py """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset_name', default='wikipedia')
    parser.add_argument('--dataset_config', default='20200501.en')
    config = parser.parse_args()

    # load datasets
    dataset = load_dataset(config.dataset_name, config.dataset_config, split='train')

    text_length = []
    for text in tqdm.tqdm(dataset['text']):
        text_length.append(text.split())

    print(f'AVG: {np.mean(text_length):.1f} MAX: {np.max(text_length):.1f}')

    plt.hist(text_length)
    plt.savefig(f'{config.dataset_name}_hist.png')


if __name__ == '__main__':
    main()
