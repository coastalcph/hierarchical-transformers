import argparse

import torch
import copy
import warnings
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
warnings.filterwarnings("ignore")


def convert_roberta_to_htf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--max_sentences', default=32)
    parser.add_argument('--num_hidden_layers', default=12)
    config = parser.parse_args()
    MAX_SENTENCE_LENGTH = 128
    MAX_SENTENCES = int(config.max_sentences)
    NUM_HIDDEN_LAYERS = int(config.num_hidden_layers)
    BERT_CHECKPOINT = 'roberta-base'

    # load pre-trained bert model and tokenizer
    roberta_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

    # load dummy config and change specifications
    roberta_config = roberta_model.config
    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # Text length parameters
    lf_config.max_position_embeddings = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES) + roberta_config.pad_token_id + 2
    lf_config.model_max_length = int(MAX_SENTENCE_LENGTH * MAX_SENTENCES)
    lf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    # Transformer parameters
    lf_config.hidden_size = roberta_config.hidden_size
    lf_config.intermediate_size = roberta_config.intermediate_size
    lf_config.num_attention_heads = roberta_config.num_attention_heads
    lf_config.hidden_act = roberta_config.hidden_act
    lf_config.attention_window = [MAX_SENTENCE_LENGTH] * NUM_HIDDEN_LAYERS
    # Vocabulary parameters
    lf_config.vocab_size = roberta_config.vocab_size
    lf_config.pad_token_id = roberta_config.pad_token_id
    lf_config.bos_token_id = roberta_config.bos_token_id
    lf_config.eos_token_id = roberta_config.eos_token_id
    lf_config.cls_token_id = tokenizer.cls_token_id
    lf_config.sep_token_id = tokenizer.sep_token_id
    lf_config.type_vocab_size = roberta_config.type_vocab_size

    # load dummy hi-transformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    lf_model.longformer.embeddings.position_embeddings.weight.data[0] = torch.zeros((roberta_config.hidden_size,))
    k = 1
    step = roberta_config.max_position_embeddings - 1
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:] = roberta_model.roberta.embeddings.position_embeddings.weight[1:(roberta_config.max_position_embeddings + 1 - k)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:(k + step)] = roberta_model.roberta.embeddings.position_embeddings.weight[1:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(roberta_model.roberta.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(roberta_model.roberta.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for i in range(len(roberta_model.roberta.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.LayerNorm)
        # attention output
        lf_model.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.dense)
        lf_model.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.LayerNorm)
        # local q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)
        # global q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)

    # copy lm_head
    lf_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())

    # check position ids
    # batch = tokenizer(['this is a dog', 'this is a cat'], return_tensors='pt')
    # lf_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

    # save model
    lf_model.save_pretrained(f'{DATA_DIR}/PLMs/longformer-roberta')

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/PLMs/longformer-roberta')

    # re-load model
    lf_model = AutoModelForMaskedLM.from_pretrained(f'{DATA_DIR}/PLMs/longformer-roberta')
    lf_tokenizer = AutoTokenizer.from_pretrained(f'{DATA_DIR}/PLMs/longformer-roberta')
    # batch = tokenizer(['this is a dog', 'this is a cat'], return_tensors='pt')
    # lf_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    print(f'RoBERTa-based Longformer model is ready to run!')


if __name__ == '__main__':
    convert_roberta_to_htf()
