export WANDB_PROJECT="hi-transformers"
export PYTHONPATH=.

MODEL_NAME='hi-transformer-p1-grouped-mlm'
MODEL_MAX_LENGTH=1024
MAX_SENTENCES=8

python language_modelling/run_pretraining_simsiam_stream.py \
    --model_name_or_path data/PLMs/${MODEL_NAME} \
    --dataset_name ./data/wikipedia-dataset \
    --dataset_config_name 20200501.en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/${MODEL_NAME}-siam \
    --overwrite_output_dir \
    --logging_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.10 \
    --weight_decay 0.01 \
    --mlm_probability 0.20 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --line_by_line \
    --pad_to_max_length \
    --sent_sim 1 \
    --doc_sim 1 \
    --mlm 1


