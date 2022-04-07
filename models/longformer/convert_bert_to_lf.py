import argparse

import torch
import copy
import warnings
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
warnings.filterwarnings("ignore")


def convert_bert_to_htf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--max_sentences', default=8)
    config = parser.parse_args()
    MAX_SENTENCE_LENGTH = 128
    MAX_SENTENCES = int(config.max_sentences)
    NUM_HIDDEN_LAYERS = 6
    BERT_LAYERS = NUM_HIDDEN_LAYERS
    BERT_CHECKPOINT = f'google/bert_uncased_L-{str(BERT_LAYERS)}_H-256_A-4'

    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

    # load dummy config and change specifications
    bert_config = bert_model.config
    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # Text length parameters
    lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * 8) + bert_config.pad_token_id
    lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    lf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    lf_config.hidden_size = bert_config.hidden_size
    lf_config.intermediate_size = bert_config.intermediate_size
    lf_config.num_attention_heads = bert_config.num_attention_heads
    lf_config.hidden_act = bert_config.hidden_act
    lf_config.attention_window = [MAX_SENTENCE_LENGTH] * NUM_HIDDEN_LAYERS
    # Vocabulary parameters
    lf_config.vocab_size = bert_config.vocab_size
    lf_config.pad_token_id = bert_config.pad_token_id
    lf_config.bos_token_id = bert_config.bos_token_id
    lf_config.eos_token_id = bert_config.eos_token_id
    lf_config.type_vocab_size = bert_config.type_vocab_size

    # load dummy hi-transformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    lf_model.longformer.embeddings.position_embeddings.weight.data[0] = torch.zeros((bert_config.hidden_size,))
    k = 1
    step = bert_config.max_position_embeddings - 1
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:] = bert_model.bert.embeddings.position_embeddings.weight[1:(bert_config.max_position_embeddings + 1 - k)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:(k + step)] = bert_model.bert.embeddings.position_embeddings.weight[1:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for i in range(len(bert_model.bert.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.LayerNorm)
        # local
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)
        # global
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)

    # copy lm_head
    lf_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
    lf_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
    lf_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
    lf_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)
    # htf_model.lm_head.load_state_dict(bert_model.lm_head.state_dict())


    # check position ids
    # batch = tokenizer(['this is a dog', 'this is a cat'], return_tensors='pt')
    # lf_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    # save model
    lf_model.save_pretrained(f'{DATA_DIR}/PLMs/longformer')

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/PLMs/longformer')

    # re-load model
    lf_model = AutoModelForMaskedLM.from_pretrained(f'{DATA_DIR}/PLMs/longformer')
    lf_tokenizer = AutoTokenizer.from_pretrained(f'{DATA_DIR}/PLMs/longformer')
    print(f'Longformer model is ready to run!')


if __name__ == '__main__':
    convert_bert_to_htf()
