import copy
import warnings

import torch
import time
from transformers import AutoConfig
import numpy as np

from data import DATA_DIR
from models.hat import HATForMaskedLM, HATConfig, HATForSequenceClassification, \
    HATForMultipleChoice, HATModelForSequentialSentenceClassification
from models.longformer import LongformerForMaskedLM, LongformerModelForSequenceClassification, LongformerForMultipleChoice , \
    LongformerModelForSentenceClassification
warnings.filterwarnings("ignore")

LAYOUTS = {
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|SD|D|D|D',
    'p1': 'S|S|SD|S|S|SD|S|S|SD|S|S|SD',
}

TASK_MODEL = {'lm': {'longformer': LongformerForMaskedLM, 'hilm': HATForMaskedLM},
              'doc_cls': {'longformer': LongformerModelForSequenceClassification, 'hilm': HATForSequenceClassification},
              'mc_qa': {'longformer': LongformerForMultipleChoice, 'hilm': HATForMultipleChoice},
              'sent_cls': {'longformer': LongformerModelForSentenceClassification, 'hilm': HATModelForSequentialSentenceClassification},
              }


def test_memory_usage(model, steps=40, batch_size=2, seq_length=4096,  mode='test', task_type='lm'):
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


def efficiency_metrics():
    for mode in ['train', 'test']:
        print(F'MODE: {mode.upper()}')
        for task in TASK_MODEL:
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
            lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
            lf_config.num_hidden_layers = 12
            # Transformer parameters
            lf_config.hidden_size = roberta_config.hidden_size
            lf_config.intermediate_size = roberta_config.intermediate_size
            lf_config.num_attention_heads = roberta_config.num_attention_heads
            # Vocabulary parameters
            lf_config.vocab_size = roberta_config.vocab_size
            lf_config.type_vocab_size = 2
            lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
            lf_config.max_sentence_length = int(MAX_SENTENCE_LENGTH)
            lf_config.max_sentences = int(MAX_SENTENCES)
            lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES) + 2
            lf_config.attention_window = [128] * roberta_config.num_hidden_layers
            lf_config.cls_token_id = 100
            lf_config.num_labels = 2
            # load dummy longformer model
            htf_model = TASK_MODEL[task]['longformer'].from_config(lf_config)
            model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
            model_total_params = model_total_params / 1e6
            memory_use, time_use = test_memory_usage(htf_model, seq_length=lf_config.model_max_length, mode=mode, task_type=task)
            lf_mem_use = copy.deepcopy(memory_use)
            lf_time_use = copy.deepcopy(time_use)
            print(f'Longformer (12-layer) model has {model_total_params:.1f}M number of parameters '
                  f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
            print('-' * 150)
            for layout in LAYOUTS:
                ENCODER_LAYOUT = {}
                for idx, block_pattern in enumerate(LAYOUTS[layout].split('|')):
                    ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                                "document_encoder": True if 'D' in block_pattern else False}

                # load dummy config and change specifications
                htf_config = HATConfig.from_pretrained(f'{DATA_DIR}/hi-transformer')
                # Text length parameters
                htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
                htf_config.MAX_SENTENCES = MAX_SENTENCES
                htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
                htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
                htf_config.num_hidden_layers = len(ENCODER_LAYOUT.keys())
                # Transformer parameters
                htf_config.hidden_size = roberta_config.hidden_size
                htf_config.intermediate_size = roberta_config.intermediate_size
                htf_config.num_attention_heads = roberta_config.num_attention_heads
                htf_config.encoder_layout = ENCODER_LAYOUT
                # Vocabulary parameters
                htf_config.vocab_size = roberta_config.vocab_size
                htf_config.type_vocab_size = 2
                lf_config.num_labels = 2
                # load dummy hi-transformer model
                htf_model = TASK_MODEL[task]['hilm'].from_config(htf_config)
                model_total_params = sum(p.numel() for p in htf_model.hat.parameters() if p.requires_grad)
                model_total_params = model_total_params / 1e6
                memory_use, time_use = test_memory_usage(htf_model, seq_length=int(MAX_SENTENCE_LENGTH * MAX_SENTENCES), mode=mode, task_type=task)
                mem_gains = (lf_mem_use / memory_use) - 1
                time_gains = (lf_time_use / time_use) - 1
                print(f'Hi-transformer model with layout {layout} has {model_total_params:.1f}M number of parameters '
                      f'{memory_use:.2f}GB peak memory use (-{mem_gains*100:.2f}%) and {time_use:.3f} batch/second (-{time_gains*100:.2f}%)!')


if __name__ == '__main__':
    efficiency_metrics()



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# MODE: TRAIN
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# TASK: SENT_CLS	NUM LAYERS: 12	NUM HIDDEN: 768	ATTENTION HEADS: 12
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# MAX SEQ LENGTH: 4096
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Longformer (12-layer) model has 148.1M number of parameters and 10.77GB peak memory use and 0.459 batch/second!
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Hi-transformer model with layout f12 has 152.2M number of parameters 8.96GB peak memory use (-20.24%) and 0.343 batch/second (-33.80%)!
# Hi-transformer model with layout p1 has 152.2M number of parameters 8.97GB peak memory use (-20.11%) and 0.344 batch/second (-33.42%)!
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# MODE: TEST
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# TASK: SENT_CLS	NUM LAYERS: 12	NUM HIDDEN: 768	ATTENTION HEADS: 12
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# MAX SEQ LENGTH: 4096
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Longformer (12-layer) model has 148.1M number of parameters and 0.98GB peak memory use and 0.131 batch/second!
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Hi-transformer model with layout f12 has 152.2M number of parameters 0.90GB peak memory use (-8.59%) and 0.115 batch/second (-13.96%)!
# Hi-transformer model with layout p1 has 152.2M number of parameters 0.90GB peak memory use (-8.59%) and 0.114 batch/second (-14.45%)!
# ------------------------------------------------------------------------------------------------------------------------------------------------------
