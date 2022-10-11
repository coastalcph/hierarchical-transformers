import copy
import warnings

import torch
import time
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from models.longformer import LongformerModelForSentenceClassification
from models.big_bird import BigBirdModelForSentenceClassification
import numpy as np
warnings.filterwarnings("ignore")


def test_memory_usage(model, steps=100, batch_size=2, seq_length=4096,  mode='test', task_type='lm'):
    model.to('cuda')
    if task_type != 'mc_qa':
        input_ids = torch.randint(1, 40000, (batch_size, seq_length), dtype=torch.long).to('cuda')
        input_ids[:, :: 128] = model.config.bos_token_id
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int).to('cuda')
    else:
        input_ids = torch.randint(1, 40000, (batch_size, 2, seq_length), dtype=torch.long).to('cuda')
        input_ids[:, :: 128] = model.config.bos_token_id
        attention_mask = torch.ones((batch_size, 2, seq_length), dtype=torch.int).to('cuda')
    if mode == 'train':
        if task_type == 'lm':
            labels = input_ids.clone()
        elif task_type == 'sent_cls':
            labels = torch.ones((batch_size, 32), dtype=torch.int).long().to('cuda')
        else:
            labels = torch.ones((batch_size, ), dtype=torch.int).long().to('cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    max_time = []
    max_mem = []
    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        if mode == 'train':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end = time.time()
        total_time = (end - start)
        max_time.append(total_time)
        max_mem.append(torch.cuda.max_memory_allocated() / 1e9)

    return np.mean(max_mem), np.mean(max_time)


def estimate_model_size():
    for mode in ['train', 'test']:
        print(F'MODE: {mode.upper()}')
        for task in ['sent_cls']:
            MAX_SENTENCE_LENGTH = 128
            roberta_config = AutoConfig.from_pretrained('roberta-base')
            print('-' * 150)
            print(F'TASK: {task.upper()}\t'
                  F'NUM LAYERS: {roberta_config.num_hidden_layers}\t'
                  F'NUM HIDDEN: {roberta_config.hidden_size}\t'
                  F'ATTENTION HEADS: {roberta_config.num_attention_heads}')
            print('-' * 150)
            MAX_SENTENCES = 32
            print('-' * 150)
            print(F'MAX SEQ LENGTH: {int(MAX_SENTENCES * MAX_SENTENCE_LENGTH)}')
            print('-' * 150)
            # load dummy longformer model
            # lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
            # lf_config.num_labels = 2
            # lf_config.max_sentence_length = 128
            # lf_config.max_sentences = 32
            # lf_config.cls_token_id = lf_config.bos_token_id
            # lf_config.sep_token_id = lf_config.eos_token_id
            # if task == 'doc_cls':
            #     htf_model = AutoModelForSequenceClassification.from_config(lf_config)
            # else:
            #     htf_model = LongformerModelForSentenceClassification.from_config(lf_config)
            # model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
            # model_total_params = model_total_params / 1e6
            # memory_use, time_use = test_memory_usage(htf_model, seq_length=4096, mode=mode,
            #                                          task_type=task)
            # print(f'Original Longformer (12-layer) model has {model_total_params:.1f}M number of parameters '
            #       f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
            # print('-' * 150)

            # load dummy bigbird model
            lf_config = AutoConfig.from_pretrained('google/bigbird-roberta-base')
            lf_config.num_labels = 2
            lf_config.max_sentence_length = 128
            lf_config.max_sentences = 32
            lf_config.cls_token_id = lf_config.bos_token_id
            lf_config.sep_token_id = lf_config.eos_token_id
            if task == 'doc_cls':
                htf_model = AutoModelForSequenceClassification.from_config(lf_config)
            else:
                htf_model = BigBirdModelForSentenceClassification.from_config(lf_config)
            model_total_params = sum(p.numel() for p in htf_model.bert.parameters() if p.requires_grad)
            model_total_params = model_total_params / 1e6
            memory_use, time_use = test_memory_usage(htf_model, seq_length=4096, mode=mode,
                                                     task_type=task)
            print(f'Original BigBird (12-layer) model has {model_total_params:.1f}M number of parameters '
                  f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
            print('-' * 150)


if __name__ == '__main__':
    estimate_model_size()
