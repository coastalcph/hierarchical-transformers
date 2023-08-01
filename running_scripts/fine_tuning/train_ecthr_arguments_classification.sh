export WANDB_PROJECT="HATs-eval"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1,2,3,4

MODEL_NAME='google/bigbird-roberta-base'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python evaluation/run_sequential_sentence_classification.py \
    --dataset_name ../data/ecthr-arguments-dataset \
    --dataset_config_name ecthr-arguments-dataset \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/${MODEL_NAME}-ecthr-args \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model micro-f1 \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --pad_to_max_length
