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

    # reduce to truncated size, max 4096
    text_length = [x if x <= 4000 else 4096 for x in text_length]

    print(f'AVG: {np.mean(text_length):.1f} ± {np.std(text_length):.1f}, MAX: {np.max(text_length):.1f}')

    # print stats in percentiles
    for min_size in [512, 1024, 2048]:
        n_docs = len([1 for x in text_length if x >= min_size])
        perc = (n_docs * 100) / len(text_length)
        print(f'No of document over {min_size} words: {n_docs}/{len(text_length)} ({perc:.1f}%)')

    # plot document length histogram
    plt.hist(text_length, range=(500, 4096), bins=50)
    plt.savefig(f'{config.dataset_name}_hist.png')


if __name__ == '__main__':
    main()
