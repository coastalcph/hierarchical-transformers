import torch
import argparse


def check_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--path', default=None)
    config = parser.parse_args()

    training_args = torch.load(config.path)
    print(training_args)


if __name__ == '__main__':
    check_args()
