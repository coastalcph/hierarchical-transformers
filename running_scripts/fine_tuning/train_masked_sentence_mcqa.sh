export WANDB_PROJECT="HATs-eval"
export PYTHONPATH=.

MODEL_NAME='hat-p1-roberta-mlm'
MODEL_MAX_LENGTH=4096
MAX_SENTENCES=32

python evaluation/run_masked_sentence_prediction.py \
    --model_name_or_path data/PLMs/${MODEL_NAME} \
    --dataset_name ./data/wikipedia-dataset \
    --dataset_config_name eval.en \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir data/PLMs/${MODEL_NAME}-mcqa-sbert \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 20 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy_score \
    --greater_is_better True \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.05 \
    --max_seq_length ${MODEL_MAX_LENGTH} \
    --max_sentences ${MAX_SENTENCES} \
    --pad_to_max_length \
    --sentence_bert_path all-MiniLM-L6-v2