#!/bin/bash
cd /root/LLaVA-MoICE
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path //hy-tmp/checkpoints/llava-v1.5-7b-MoICE-textvqa \
    --question-file /hy-tmp/TextVQA/llava_v1_5_4k.jsonl \
    --image-folder /hy-tmp/TextVQA/val_images \
    --answers-file /hy-tmp/TextVQA/llava-v1.5-7b_answers.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /hy-tmp/TextVQA/TextVQA_0.5.1_val.jsonl \
    --result-file /hy-tmp/TextVQA/llava-v1.5-7b_answers.jsonl
