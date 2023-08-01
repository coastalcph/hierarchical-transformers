export WANDB_PROJECT="HATs-eval"
export PYTHONPATH=.

MODEL_NAME='allenai/longformer-base-4096'
POOLING_METHOD='last'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python evaluation/run_quality_mcqa.py \
    --model_name_or_path ${MODEL_NAME} \
    --pooling ${POOLING_METHOD} \
    --dataset_name data/quality-dataset \
    --dataset_config_name quality \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/${MODEL_NAME}-${POOLING_METHOD}-quality \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy_score \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --pad_to_max_length