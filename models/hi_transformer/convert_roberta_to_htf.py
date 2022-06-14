import argparse

import warnings
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerConfig, HiTransformerTokenizer
warnings.filterwarnings("ignore")

LAYOUTS = {
    'p1': 'S|S|SD|S|S|SD|S|S|SD|S|S|SD',
    'l1': 'S|S|S|S|S|SD|S|SD|S|SD|S|SD',
    'f12': 'S|S|S|S|S|S|S|S|S|S|S|SD|D|D|D'
}


def convert_roberta_to_htf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--layout', default='p1', choices=['p1', 'l1', 'f12'],
                        help='S|D encoders layout')
    parser.add_argument('--max_sentences', default=32)
    config = parser.parse_args()
    MAX_SENTENCE_LENGTH = 128
    MAX_SENTENCES = int(config.max_sentences)
    ENCODER_LAYOUT = {}
    for idx, block_pattern in enumerate(LAYOUTS[config.layout].split('|')):
        ENCODER_LAYOUT[str(idx)] = {"sentence_encoder": True if 'S' in block_pattern else False,
                                    "document_encoder": True if 'D' in block_pattern else False}

    NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())
    ROBERTA_CHECKPOINT = 'roberta-base'

    # load pre-trained bert model and tokenizer
    roberta_model = AutoModelForMaskedLM.from_pretrained(ROBERTA_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

    # load dummy config and change specifications
    roberta_config = roberta_model.config
    htf_config = HiTransformerConfig.from_pretrained(f'{DATA_DIR}/hi-transformer')
    # Text length parameters
    htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
    htf_config.max_sentences = MAX_SENTENCES
    htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH + 2
    htf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    htf_config.hidden_size = roberta_config.hidden_size
    htf_config.intermediate_size = roberta_config.intermediate_size
    htf_config.num_attention_heads = roberta_config.num_attention_heads
    htf_config.hidden_act = roberta_config.hidden_act
    htf_config.encoder_layout = ENCODER_LAYOUT
    # Vocabulary parameters
    htf_config.vocab_size = roberta_config.vocab_size
    htf_config.pad_token_id = roberta_config.pad_token_id
    htf_config.bos_token_id = roberta_config.bos_token_id
    htf_config.eos_token_id = roberta_config.eos_token_id
    htf_config.type_vocab_size = roberta_config.type_vocab_size

    # load dummy hi-transformer model
    htf_model = HiTransformerForMaskedLM.from_config(htf_config)

    # copy embeddings
    htf_model.hi_transformer.embeddings.position_embeddings.weight.data = roberta_model.roberta.embeddings.position_embeddings.weight[:MAX_SENTENCE_LENGTH+roberta_config.pad_token_id+1]
    htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(roberta_model.roberta.embeddings.word_embeddings.state_dict())
    htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
    htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(roberta_model.roberta.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for idx in range(min(NUM_HIDDEN_LAYERS, roberta_config.num_hidden_layers)):
        if htf_model.config.encoder_layout[str(idx)]['sentence_encoder']:
            htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(roberta_model.roberta.encoder.layer[idx].state_dict())
        if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
            htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(roberta_model.roberta.encoder.layer[idx].state_dict())
            htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = roberta_model.roberta.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]

    # copy lm_head
    htf_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())

    # save model
    htf_model.save_pretrained(f'{DATA_DIR}/PLMs/hi-transformer-{config.layout}-roberta')

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/PLMs/hi-transformer-{config.layout}-roberta')

    # re-load model
    htf_model = HiTransformerForMaskedLM.from_pretrained(f'{DATA_DIR}/PLMs/hi-transformer-{config.layout}-roberta')
    htf_tokenizer = HiTransformerTokenizer.from_pretrained(f'{DATA_DIR}/PLMs/hi-transformer-{config.layout}-roberta')
    print(f'RoBERTa-based Hi-transformer model with layout {config.layout} is ready to run!')

    # input_ids = torch.randint(1, 30000, (2, 1024), dtype=torch.long)
    # input_ids[:, :: 128] = htf_tokenizer.cls_token_id
    # labels = input_ids.clone()
    # attention_mask = torch.ones((2, 1024), dtype=torch.int)
    # htf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # roberta_model(input_ids=input_ids[:, :128], attention_mask=attention_mask[:, :128], labels=labels[:, :128])


if __name__ == '__main__':
    convert_roberta_to_htf()
