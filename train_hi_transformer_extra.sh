export WANDB_PROJECT="hi-transformers"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

LAYOUT='s1'
MODEL_WARMUP_STRATEGY='grouped'
MODEL_MAX_LENGTH=1024
MAX_SENTENCES=8

python3 language_modelling/xla_spawn.py --num_cores=8 language_modelling/run_pretraining.py \
    --model_name_or_path data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY}-mlm \
    --dataset_name ./data/wikipedia-dataset \
    --dataset_config_name 20200501.en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY}-mlm+srp_embed \
    --overwrite_output_dir \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.20 \
    --ms_probability 0.25 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length \
    --srp 1 \
    --mlm 1 \
    --sentence_bert_path all-MiniLM-L6-v2


