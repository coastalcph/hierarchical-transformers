export WANDB_PROJECT="hi-transformers"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export PYTHONPATH=.

XLA_IR_DEBUG=1 python3 language_modelling/xla_spawn.py --num_cores=8 language_modelling/run_mlm_stream.py \
    --config_name data/hi-transformer \
    --dataset_name ./data/wikipedia-dataset \
    --dataset_config_name 20200501.en \
    --do_train \
    --do_eval \
    --output_dir data/PLMs/hi-transformer-test \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 5 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --mlm_probability 0.2 \
    --max_seq_length 1024 \
    --line_by_line \
    --pad_to_max_length \
    --max_train_samples 512 \
    --max_eval_samples 512