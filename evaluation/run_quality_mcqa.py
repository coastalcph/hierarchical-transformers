#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning sentence classification models"""
import copy
import logging
import os
import random
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForMultipleChoice,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from models.hat import HATConfig, HATTokenizer, HATForMultipleChoice
from models.longformer import LongformerTokenizer, LongformerForMultipleChoice
from sklearn.metrics import accuracy_score
from language_modelling.data_collator import DataCollatorForMultipleChoice

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_sentences: int = field(
        default=32,
        metadata={
            "help": "The maximum number of sentences after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    sentence_bert_path: Optional[str] = field(
        default=None, #'all-MiniLM-L6-v2',
        metadata={
            "help": "The name of the sentence-bert to use (via the transforrmers library)"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    pooling: str = field(
        default='last', metadata={"help": "Which pooling method to use (first or last token)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        train_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="train",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_eval:
        eval_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="validation",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    if training_args.do_predict:
        predict_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="test",
            data_dir=data_args.dataset_name,
            cache_dir=model_args.cache_dir,
        )

    # Labels
    num_labels = 4
    random.seed(training_args.seed)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'hat' in model_args.model_name_or_path:
        config = HATConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="quality",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = HATTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = HATForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif "allenai/longformer-base-4096" in model_args.model_name_or_path or "google/bigbird-roberta-base" in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="document-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="sentence-mcqa",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.max_sentence_size = 128
        config.max_sentence_length = 128
        config.max_sentences = data_args.max_sentences
        tokenizer = LongformerTokenizer.from_pretrained(
            model_args.model_name_or_path,
            do_lower_case=model_args.do_lower_case,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = LongformerForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        batch = {'input_ids': [], 'attention_mask': [], 'label': []}
        if 'allenai/longformer' in model_args.model_name_or_path:
            batch['global_attention_mask'] = []
        batch_articles = tokenizer(
            [re.sub('[\n ]{2,}', '\n', article) for article in examples["article"]],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        batch_articles = tokenizer.pad(
            batch_articles,
            padding=padding,
            max_length=data_args.max_seq_length,
            pad_to_multiple_of=data_args.max_seq_length,
        )

        batch_question = tokenizer(
            examples["question"],
            padding=padding,
            max_length=128,
            truncation=True,
        )

        for example_idx, options in enumerate(examples['options']):
            batch_options = tokenizer(
                options,
                padding=padding,
                max_length=128,
                truncation=True,
            )

            for option_idx, _ in enumerate(options):
                batch['input_ids'].append(copy.deepcopy(batch_articles['input_ids'][example_idx]))
                batch['input_ids'][-1][-256:-128] = batch_question['input_ids'][example_idx][:128]
                batch['input_ids'][-1][-128:] = batch_options['input_ids'][option_idx][:128]
                if 'longformer' in model_args.model_name_or_path or 'bigbird' in model_args.model_name_or_path:
                    batch['input_ids'][-1][-256] = tokenizer.sep_token_id
                    batch['input_ids'][-1][-128] = tokenizer.sep_token_id
                if 'allenai/longformer' in model_args.model_name_or_path:
                    batch['global_attention_mask'].append([0] * data_args.max_seq_length)
                    batch['global_attention_mask'][-1][0] = 1
                    batch['global_attention_mask'][-1][-256] = 1
                    batch['global_attention_mask'][-1][-128] = 1
                batch['attention_mask'].append(copy.deepcopy(batch_articles['attention_mask'][example_idx]))
                batch['attention_mask'][-1][-256:-128] = batch_question['attention_mask'][example_idx][:128]
                batch['attention_mask'][-1][-128:] = batch_options['attention_mask'][option_idx][:128]
                batch['label'].append(examples['label'][example_idx]-1)

        return {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in batch.items()}

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        acc = accuracy_score(y_true=p.label_ids, y_pred=preds)

        return {'accuracy_score': acc}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = DataCollatorForMultipleChoice(tokenizer)
    elif training_args.fp16:
        data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predictions.csv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()