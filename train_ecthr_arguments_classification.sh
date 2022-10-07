export WANDB_PROJECT="hi-transformers-eval"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=7

MODEL_NAME='hi-transformer-p1-roberta-mlm'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python evaluation/run_sequential_sentence_classification.py \
    --model_name_or_path data/PLMs/${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/${MODEL_NAME}-${POOLING_METHOD}-ecthr-args \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --pad_to_max_length














