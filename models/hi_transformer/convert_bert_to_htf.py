from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerForSequenceClassification, \
    HiTransformerConfig, HiTransformerTokenizer

MAX_SENTENCE_LENGTH = 128
MAX_SENTENCES = 64
ENCODER_LAYOUT = {
    "0": {"sentence_encoder": True, "document_encoder":  False},
    "1": {"sentence_encoder": True, "document_encoder":  False},
    "2": {"sentence_encoder": True, "document_encoder":  True},
    "3": {"sentence_encoder": True, "document_encoder":  False},
    "4": {"sentence_encoder": True, "document_encoder":  False},
    "5": {"sentence_encoder": True, "document_encoder":  True}}
NUM_HIDDEN_LAYERS = len(ENCODER_LAYOUT.keys())
BERT_CHECKPOINT = 'google/bert_uncased_L-6_H-256_A-8'

# load pre-trained bert model and tokenizer
bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=MAX_SENTENCE_LENGTH * MAX_SENTENCES)

# load dummy config and change specifications
bert_config = bert_model.config
htf_config = HiTransformerConfig.from_pretrained('../../data/hi-transformer')
htf_config.max_sentence_length = MAX_SENTENCE_LENGTH
htf_config.max_sentences = MAX_SENTENCES
htf_config.num_hidden_layers = NUM_HIDDEN_LAYERS
htf_config.hidden_size = bert_config.hidden_size
htf_config.intermediate_size = bert_config.intermediate_size
htf_config.max_position_embeddings = MAX_SENTENCE_LENGTH
htf_config.model_max_length = MAX_SENTENCE_LENGTH * MAX_SENTENCES
htf_config.vocab_size = bert_config.vocab_size
htf_config.pad_token_id = bert_config.pad_token_id
htf_config.bos_token_id = bert_config.bos_token_id
htf_config.eos_token_id = bert_config.eos_token_id
htf_config.type_vocab_size = bert_config.type_vocab_size


# load dummy hi-transformer model
htf_model = HiTransformerForMaskedLM.from_config(htf_config)

# copy embeddings
htf_model.hi_transformer.embeddings.position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[:MAX_SENTENCE_LENGTH+htf_config.pad_token_id+1]
htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

# copy transformer layers
for idx in range(NUM_HIDDEN_LAYERS):
    htf_model.hi_transformer.encoder.layer[idx].sentence_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
    if htf_model.config.encoder_layout[str(idx)]['document_encoder']:
        htf_model.hi_transformer.encoder.layer[idx].document_encoder.load_state_dict(bert_model.bert.encoder.layer[idx].state_dict())
        htf_model.hi_transformer.encoder.layer[idx].position_embeddings.weight.data = bert_model.bert.embeddings.position_embeddings.weight[1:MAX_SENTENCES+2]

# copy lm_head
htf_model.lm_head.load_state_dict(bert_model.lm_head.state_dict())

# save model
htf_model.save_pretrained('../../data/PLMs/hi-transformer-bert')

# save tokenizer
htf_tokenizer = HiTransformerTokenizer.from_pretrained('../../data/hi-transformer')
htf_tokenizer._tokenizer = tokenizer
htf_tokenizer.save_pretrained('../../data/PLMs/hi-transformer-bert')

# re-load model
htf_model = HiTransformerForSequenceClassification.from_pretrained('../../data/PLMs/hi-transformer-bert')
print()
