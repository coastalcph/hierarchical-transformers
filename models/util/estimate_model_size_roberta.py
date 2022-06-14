import copy
import warnings

import torch
import time
from transformers import AutoConfig, AutoModelForMaskedLM

from data import DATA_DIR
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerConfig
from models.longformer import LongformerForMaskedLM

warnings.filterwarnings("ignore")

LAYOUTS = {
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|SD|D|D|D',
    'p1': 'S|S|SD|S|S|SD|S|S|SD|S|S|SD',
    'l1': 'S|S|S|S|S|SD|S|SD|S|SD|S|SD',
}


def test_memory_usage(model, steps=10, batch_size=2, seq_length=1024):
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')
    input_ids = torch.randint(1, 30000, (batch_size, seq_length), dtype=torch.long).to('cuda')
    input_ids[:, :: 128] = model.config.bos_token_id
    labels = input_ids.clone()
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    start = time.time()
    for _ in range(steps):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    end = time.time()
    total_time = (end - start) / steps
    return torch.cuda.max_memory_allocated() / 1e9, total_time


def estimate_model_size():
    MAX_SENTENCE_LENGTH = 128
    roberta_config = AutoConfig.from_pretrained('roberta-base')
    print('-' * 150)
    print(F'NUM LAYERS: {roberta_config.num_hidden_layers}\t'
          F'NUM HIDDEN: {roberta_config.hidden_size}\t'
          F'ATTENTION HEADS: {roberta_config.num_attention_heads}')
    print('-' * 150)
    MAX_SENTENCES = 32
    print('-' * 150)
    print(F'MAX SEQ LENGTH: {int(MAX_SENTENCES * MAX_SENTENCE_LENGTH)}')
    print('-' * 150)
    # load dummy longformer model
    htf_model = AutoModelForMaskedLM.from_config(roberta_config)
    model_total_params = sum(p.numel() for p in htf_model.roberta.parameters() if p.requires_grad)
    model_total_params = model_total_params / 1e6
    memory_use, time_use = test_memory_usage(htf_model, seq_length=512)
    print(f'RoBERTa model has {model_total_params:.1f}M number of parameters '
          f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
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
    lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES) + 2
    lf_config.attention_window = [128] * roberta_config.num_hidden_layers
    lf_config.cls_token_id = 100
    # load dummy longformer model
    htf_model = LongformerForMaskedLM.from_config(lf_config)
    model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
    model_total_params = model_total_params / 1e6
    memory_use, time_use = test_memory_usage(htf_model, seq_length=lf_config.model_max_length)
    lf_mem_use = copy.deepcopy(memory_use)
    lf_time_use = copy.deepcopy(time_use)
    print(f'Longformer (12-layer) model has {model_total_params:.1f}M number of parameters '
          f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
    print('-' * 150)

    # lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # lf_config.num_hidden_layers = 10
    # # Transformer parameters
    # lf_config.hidden_size = roberta_config.hidden_size
    # lf_config.intermediate_size = roberta_config.intermediate_size
    # lf_config.num_attention_heads = roberta_config.num_attention_heads
    # # Vocabulary parameters
    # lf_config.vocab_size = roberta_config.vocab_size
    # lf_config.type_vocab_size = 2
    # lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    # lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES) + 2
    # lf_config.attention_window = [128] * 10
    # lf_config.cls_token_id = 100
    # # load dummy longformer model
    # htf_model = LongformerForMaskedLM.from_config(lf_config)
    # model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
    # model_total_params = model_total_params / 1e6
    # memory_use, time_use = test_memory_usage(htf_model, seq_length=lf_config.model_max_length)
    # print(f'Longformer (10-layer) model has {model_total_params:.1f}M number of parameters '
    #        f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
    # print('-' * 150)
    #
    # lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # lf_config.num_hidden_layers = 8
    # # Transformer parameters
    # lf_config.hidden_size = roberta_config.hidden_size
    # lf_config.intermediate_size = roberta_config.intermediate_size
    # lf_config.num_attention_heads = roberta_config.num_attention_heads
    # # Vocabulary parameters
    # lf_config.vocab_size = roberta_config.vocab_size
    # lf_config.type_vocab_size = 2
    # lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    # lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES) + 2
    # lf_config.attention_window = [128] * 8
    # lf_config.cls_token_id = 100
    # # load dummy longformer model
    # htf_model = LongformerForMaskedLM.from_config(lf_config)
    # model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
    # model_total_params = model_total_params / 1e6
    # memory_use, time_use = test_memory_usage(htf_model, seq_length=lf_config.model_max_length)
    # lf_mem_use = copy.deepcopy(memory_use)
    # lf_time_use = copy.deepcopy(time_use)
    # print(f'Longformer (8-layer) model has {model_total_params:.1f}M number of parameters '
    #       f'and {memory_use:.2f}GB peak memory use and {time_use:.3f} batch/second!')
    # print('-' * 150)


    for layout in LAYOUTS:
        ENCODER_LAYOUT = {}
        for idx, block_pattern in enumerate(LAYOUTS[layout].split('|')):
            ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                        "document_encoder": True if 'D' in block_pattern else False}

        # load dummy config and change specifications
        htf_config = HiTransformerConfig.from_pretrained(f'{DATA_DIR}/hi-transformer')
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

        # load dummy hi-transformer model
        htf_model = HiTransformerForMaskedLM.from_config(htf_config)
        model_total_params = sum(p.numel() for p in htf_model.hi_transformer.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        memory_use, time_use = test_memory_usage(htf_model, seq_length=int(MAX_SENTENCE_LENGTH * MAX_SENTENCES))
        mem_gains = (lf_mem_use / memory_use) - 1
        time_gains = (lf_time_use / time_use) - 1
        print(f'Hi-transformer model with layout {layout} has {model_total_params:.1f}M number of parameters '
              f'{memory_use:.2f}GB peak memory use (-{mem_gains*100:.2f}%) and {time_use:.3f} batch/second (-{time_gains*100:.2f}%)!')


if __name__ == '__main__':
    estimate_model_size()
