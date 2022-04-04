export WANDB_PROJECT="hi-transformers"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

LAYOUT='s1'
MODEL_WARMUP_STRATEGY='grouped'

python3 models/hi_transformer/convert_bert_to_htf.py --layout ${LAYOUT} --warmup_strategy ${MODEL_WARMUP_STRATEGY}

python3 language_modelling/xla_spawn.py --num_cores=8 language_modelling/run_mlm.py \
    --config_name data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY} \
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY}-mlm \
    --overwrite_output_dir \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_steps 50000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --lr_scheduler_type linear_schedule_with_warmup \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length 1024 \
    --line_by_line \
    --pad_to_max_length