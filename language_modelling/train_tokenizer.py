from tokenizers import models, normalizers, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from datasets import load_dataset
import os
import argparse

CUSTOM_TOK_FOLDER = '../data/custom-tokenizer'


def batch_iterator(dataset):
    for example in dataset['text']:
        yield example


def main():
    """ set default hyperparams in default_hyperparams.py """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--vocab_size', default=50000)
    config = parser.parse_args()

    # configure tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                            add_prefix_space=True, trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )

    # load datasets
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')

    # train tokenizer
    tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(dataset))

    # save tokenizer
    tokenizer.save(os.path.join(CUSTOM_TOK_FOLDER, 'tokenizer.json'), pretty=True)
    tokenizer.model.save(CUSTOM_TOK_FOLDER)


if __name__ == '__main__':
    main()
