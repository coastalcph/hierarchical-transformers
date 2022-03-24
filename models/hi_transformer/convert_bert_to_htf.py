from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerForSequenceClassification, \
    HiTransformerConfig, HiTransformerTokenizer

MAX_POSITION = 8192
MAX_SENTENCE_LENGTH = 128
MAX_SENTENCES = 64
NUM_HIDDEN_LAYERS = 6


# load pre-trained bert model and tokenizer
roberta_model = AutoModelForMaskedLM.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=MAX_POSITION)

# load dummy config and change specifications
roberta_config = roberta_model.config
htf_config = HiTransformerConfig.from_pretrained('../../data/hi-transformer-v2')
htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
htf_config.max_sentences = MAX_SENTENCES
htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
htf_config.hidden_size = roberta_config.hidden_size
htf_config.intermediate_size = roberta_config.intermediate_size
htf_config.max_position_embeddings = 8192
htf_config.vocab_size = roberta_config.vocab_size
htf_config.pad_token_id = roberta_config.pad_token_id
htf_config.bos_token_id = roberta_config.bos_token_id
htf_config.eos_token_id = roberta_config.eos_token_id
htf_config.type_vocab_size = roberta_config.type_vocab_size


# load dummy hi-transformer model
htf_model = HiTransformerForMaskedLM.from_config(htf_config)

# copy embeddings
htf_model.hi_transformer.embeddings.position_embeddings.weight.data = roberta_model.roberta.embeddings.position_embeddings.weight[:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(roberta_model.roberta.embeddings.word_embeddings.state_dict())
htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(roberta_model.roberta.embeddings.LayerNorm.state_dict())

# copy transformer layers
for idx in range(NUM_HIDDEN_LAYERS):
    htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(roberta_model.roberta.encoder.layer[idx].state_dict())
    if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
        htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(roberta_model.roberta.encoder.layer[idx].state_dict())
        htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = roberta_model.roberta.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]

# copy lm_head
htf_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())

# save model
htf_model.save_pretrained('../../data/PLMs/hi-transformer-roberta')

# save tokenizer
htf_tokenizer = HiTransformerTokenizer.from_pretrained('roberta-base')
htf_tokenizer._tokenizer = tokenizer
htf_tokenizer.save_pretrained('../../data/PLMs/hi-transformer-v2-roberta')

# re-load model
htf_model = HiTransformerForSequenceClassification.from_pretrained('../../data/PLMs/hi-transformer-v2-roberta')
print()
