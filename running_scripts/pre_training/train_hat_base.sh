export WANDB_PROJECT="HATs-pretrain"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

LAYOUT='p1'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python3 models/hat/convert_roberta_to_htf.py --layout ${LAYOUT} --max_sentences ${MAX_SENTENCES}

python3 language_modelling/xla_spawn.py --num_cores=8 language_modelling/run_mlm_stream.py \
    --model_name_or_path data/PLMs/hat-${LAYOUT}-roberta \
    --dataset_name c4 \
    --dataset_config_name en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/hi-transformer-${LAYOUT}-roberta-mlm \
    --overwrite_output_dir \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 5 \
    --max_steps 50000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --min_sequence_length 1024 \
    --pad_to_max_length \
    --max_eval_samples 100000