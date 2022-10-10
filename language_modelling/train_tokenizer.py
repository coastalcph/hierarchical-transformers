from tokenizers import models, normalizers, pre_tokenizers, decoders, processors, trainers
from tokenizers import Tokenizer
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import argparse

CUSTOM_TOK_FOLDER = '../data/custom-tokenizer'
hat_FOLDER = '../data/hi-transformer'
ROBERTA_FOLDER = '../data/roberta'


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
    backend_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    backend_tokenizer.normalizer = normalizers.Lowercase()
    backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    backend_tokenizer.decoder = decoders.ByteLevel()
    backend_tokenizer.post_processor = processors.RobertaProcessing(sep=("</s>", 2), cls=("<s>", 1),
                                                                    add_prefix_space=True, trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True
    )

    # load datasets
    dataset = load_dataset("multi_eurlex", "en", split='train')

    # train tokenizer
    backend_tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(dataset))

    # test tokenizer
    tokens = backend_tokenizer.encode('dog ' * 5, add_special_tokens=False)
    print('Original Tokenizer: ', tokens.tokens)

    # save tokenizer
    new_roberta_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_max_length=512,
        # padding_side="Set me if you want",
        # truncation_side="Set me if you want",
        # model_input_names="Set me if you want",
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        sep_token='</s>',
        pad_token='<pad>',
        cls_token='<s>',
        mask_token='<mask>',
    )

    new_hat_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tokenizer,
        model_max_length=8192,
        # padding_side="Set me if you want",
        # truncation_side="Set me if you want",
        # model_input_names="Set me if you want",
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        sep_token='</s>',
        pad_token='<pad>',
        cls_token='<s>',
        mask_token='<mask>',
    )

    new_roberta_tokenizer.save_pretrained(CUSTOM_TOK_FOLDER)
    new_roberta_tokenizer.save_pretrained(ROBERTA_FOLDER)
    new_hat_tokenizer.save_pretrained(hat_FOLDER)

    # re-load tokenizer and test
    reloaded_tokenizer = AutoTokenizer.from_pretrained(CUSTOM_TOK_FOLDER)
    tokens = reloaded_tokenizer.tokenize('dog ' * 5, add_special_tokens=False)
    print('Reloaded Tokenizer: ', tokens)


if __name__ == '__main__':
    main()
