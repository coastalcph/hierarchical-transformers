export WANDB_PROJECT="HATs-pretrain"
export PYTHONPATH=.

MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python models/longformer/convert_roberta_to_lf.py --max_sentences ${MAX_SENTENCES} --num_hidden_layers 12

python language_modelling/run_mlm_stream.py \
    --model_name_or_path data/PLMs/longformer-roberta \
    --dataset_name c4 \
    --dataset_config_name en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/longformer-roberta-mlm \
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
    --gradient_accumulation_steps 16 \
    --eval_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.15 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --min_sequence_length 1024 \
    --pad_to_max_length \
    --max_eval_samples 100000