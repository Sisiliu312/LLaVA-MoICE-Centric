#!/bin/bash
cd /root/LLaVA-MoICE
export PYTHONPATH=/root/LLaVA-MoICE

deepspeed llava/train/train_mem.py \
    --deepspeed /root/LLaVA-MoICE/scripts/v1_5/ds_z3_bf16.json \
    --model_name_or_path /hy-tmp/llava-v1.5-7b \
    --version v1 \
    --data_path /hy-tmp/TextVQA/llava_v1_5_1k_train.json \
    --image_folder /hy-tmp/TextVQA/val_images \
    --vision_tower /hy-tmp/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /hy-tmp/checkpoints/test \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --logging_steps 1 \
    --lr_scheduler_type constant \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --router_aux_loss_coef 0.3 \
    --pretrain_loss True \
    --topk 7 \
    --expert_nums 7 \
    --base_set "[10000,17500,18000,19000,20000,22500,25000]" \
    --only_train_gate True \

# if [ $? -eq 0 ]; then
#     echo "Training completed successfully. Shutting down..."
#     sudo shutdown now
# else
#     echo "Training failed."
#     sudo shutdown now
# fi
