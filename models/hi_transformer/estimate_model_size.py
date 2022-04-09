import warnings
import torch
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoConfig
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerConfig
warnings.filterwarnings("ignore")

LAYOUTS = {
    's1': 'SD|SD|SD|SD|SD|SD',
    's2': 'S|SD|D|S|SD|D|S|SD|D',
    'p1': 'S|SD|S|SD|S|SD|S|SD',
    'p2': 'S|S|SD|S|S|SD|S|S|SD',
}


def test_memory_usage(model):
    torch.cuda.reset_peak_memory_stats()
    model.to('cuda')
    input_ids = torch.zeros((1, 1024), dtype=torch.int).to('cuda')
    attention_mask = torch.ones((1, 1024), dtype=torch.int).to('cuda')
    optimizer = torch.optim.AdamW(lr=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e-4)
    for step, batch in range(10):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return torch.cuda.max_memory_allocated() / 1e9


def estimate_model_size():

    MAX_SENTENCE_LENGTH = 128
    MAX_SENTENCES = 8
    BERT_CHECKPOINT = f'google/bert_uncased_L-6_H-256_A-4'

    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    for layout in ['s1', 's2', 'p1', 'p2']:
        ENCODER_LAYOUT = {}
        for idx, block_pattern in enumerate(LAYOUTS[layout].split('|')):
            ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                        "document_encoder": True if 'D' in block_pattern else False}

        NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())
        # load dummy config and change specifications
        bert_config = bert_model.config
        htf_config = HiTransformerConfig.from_pretrained(f'{DATA_DIR}/hi-transformer')
        # Text length parameters
        htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
        htf_config.max_sentences = MAX_SENTENCES
        htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
        htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
        htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
        # Transformer parameters
        htf_config.hidden_size = bert_config.hidden_size
        htf_config.intermediate_size = bert_config.intermediate_size
        htf_config.num_attention_heads = bert_config.num_attention_heads
        htf_config.encoder_layout = ENCODER_LAYOUT
        # Vocabulary parameters
        htf_config.vocab_size = bert_config.vocab_size
        htf_config.type_vocab_size = bert_config.type_vocab_size

        # load dummy hi-transformer model
        htf_model = HiTransformerForMaskedLM.from_config(htf_config)
        model_total_params = sum(p.numel() for p in htf_model.hi_transformer.parameters() if p.requires_grad)
        model_total_params = model_total_params / 1e6
        memory_use = test_memory_usage(htf_model)
        print(f'Hi-transformer model with layout {layout} has {model_total_params:.1f}M number of parameters '
              f'and {memory_use:.1f}GB peak memory use!')

    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    lf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    lf_config.hidden_size = bert_config.hidden_size
    lf_config.intermediate_size = bert_config.intermediate_size
    lf_config.num_attention_heads = bert_config.num_attention_heads
    # Vocabulary parameters
    lf_config.vocab_size = bert_config.vocab_size
    lf_config.type_vocab_size = bert_config.type_vocab_size
    lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * 8)
    lf_config.attention_window = [128] * NUM_HIDDEN_LAYERS
    # load dummy longformer model
    htf_model = AutoModelForMaskedLM.from_config(lf_config)
    model_total_params = sum(p.numel() for p in htf_model.longformer.parameters() if p.requires_grad)
    model_total_params = model_total_params / 1e6
    memory_use = test_memory_usage(htf_model)
    print(f'Longformer model has {model_total_params:.1f}M number of parameters '
          f'and {memory_use:.1f}GB peak memory use!')


if __name__ == '__main__':
    estimate_model_size()
