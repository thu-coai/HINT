env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3.7 -u finetune.py \
    --data_dir ../Data/data/bc \
    --output_dir=./bart_post_bc \
    --save_top_k 80 \
    --train_batch_size=10 \
    --eval_batch_size=10 \
    --num_train_epochs 10 \
    --model_name_or_path ./bart_model \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --n_val 1000 \
    --val_check_interval 0.1 \
    --overwrite_output_dir \
    --sbert \
    --reorder \