from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.hi_transformer import HiTransformerForMaskedLM, HiTransformerConfig, HiTransformerTokenizer
import copy

max_pos = 8192
max_sentence_length = 128
max_sentences = 64

# load pre-trained bert model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", model_max_length=max_pos)

# load dummy config and change specifications
config = model.config
htf_config = HiTransformerConfig.from_pretrained('../data/hi-transformer')
htf_config.num_hidden_layers = config.num_hidden_layers
htf_config.max_sentence_length = max_sentence_length
htf_config.max_sentences = max_sentences
htf_config.max_position_embeddings = 8192

# load dummy hi-transformer model
htf_model = HiTransformerForMaskedLM.from_config(htf_config)

# copy embeddings
htf_model.hi_transformer.embeddings.position_embeddings.weight.data = model.bert.embeddings.position_embeddings.weight[:max_sentence_length]
htf_model.hi_transformer.embeddings.word_embeddings.load_state_dict(model.bert.embeddings.word_embeddings.state_dict())
htf_model.hi_transformer.embeddings.token_type_embeddings.load_state_dict(model.bert.embeddings.token_type_embeddings.state_dict())
htf_model.hi_transformer.embeddings.LayerNorm.load_state_dict(model.bert.embeddings.LayerNorm.state_dict())

# copy transformer layers
for i in range(len(model.bert.encoder.layer)):
    htf_model.hi_transformer.encoder.layer[i].position_embeddings.weight.data = model.bert.embeddings.position_embeddings.weight[:max_sentences+1]
    htf_model.hi_transformer.encoder.layer[i].sentence_encoder.load_state_dict(model.bert.encoder.layer[i].state_dict())
    htf_model.hi_transformer.encoder.layer[i].document_encoder.load_state_dict(model.bert.encoder.layer[i].state_dict())

# copy lm_head
htf_model.lm_head.dense.load_state_dict(model.cls.predictions.transform.dense.state_dict())
htf_model.lm_head.layer_norm.load_state_dict(model.cls.predictions.transform.LayerNorm.state_dict())
htf_model.lm_head.decoder.load_state_dict(model.cls.predictions.decoder.state_dict())
htf_model.lm_head.bias = copy.deepcopy(model.cls.predictions.bias)

# save model
htf_model.save_pretrained('../data/hi-transformer-wu')

# save tokenizer
htf_tokenizer = HiTransformerTokenizer.from_pretrained('../data/hi-transformer')
htf_tokenizer._tokenizer = tokenizer
htf_tokenizer.save_pretrained('../data/hi-transformer-wu')

# re-load model
htf_model = HiTransformerForMaskedLM.from_pretrained('../data/hi-transformer-wu')
