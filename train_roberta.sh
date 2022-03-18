#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH -p gpu --gres=gpu:titanrtx:2 --mem=32GB
#SBATCH --time=60:00:00
#SBATCH --output=roberta-v1.txt
#SBATCH --job-name=roberta-v1

hostname
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.

BATCH_SIZE=64
ACCUMULATION_STEPS=8

python language_modelling/run_mlm.py \
    --config_name data/roberta \
    --dataset_name multi_eurlex \
    --dataset_config_name en \
    --do_train 1 \
    --do_eval 1 \
    --output_dir data/PLMs/roberta \
    --overwrite_output_dir 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 20 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --fp16 \
    --fp16_full_eval \
    --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${ACCUMULATION_STEPS} \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_seq_length 512 \
    --max_train_samples 32 \
    --max_eval_samples 32
