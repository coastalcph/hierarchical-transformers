export WANDB_PROJECT="HATs-eval"
export PYTHONPATH=.

MODEL_NAME='hat-s1-grouped-mlm'
POOLING_METHOD='max'
MODEL_MAX_LENGTH=1024
MAX_SENTENCES=8

python evaluation/run_document_classification.py \
    --model_name_or_path data/PLMs/${MODEL_NAME} \
    --pooling ${POOLING_METHOD} \
    --dataset_name multi_eurlex \
    --dataset_config_name en \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/${MODEL_NAME}-${POOLING_METHOD}-dc \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model micro_f1 \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --pad_to_max_length