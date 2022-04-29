export WANDB_PROJECT="hi-transformers-eval"
export PYTHONPATH=.

LAYOUT='s1'
MODEL_WARMUP_STRATEGY='grouped'
MODEL_TYPE='mlm'
MODEL_MAX_LENGTH=1024
MAX_SENTENCES=8

python evaluation/run_sentence_order.py \
    --model_name_or_path data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY}-${MODEL_TYPE} \
    --dataset_name ./data/wikipedia-dataset \
    --dataset_config_name 20200501.en \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/hi-transformer-${LAYOUT}-${MODEL_WARMUP_STRATEGY}-${MODEL_TYPE}-sop \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy_score \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --pad_to_max_length