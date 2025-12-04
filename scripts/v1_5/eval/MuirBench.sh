#!/bin/bash
cd /root/LLaVA-MoICE
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /hy-tmp/checkpoints/llava-v1.5-7b-v2-centric \
    --question-file /hy-tmp/MuirBench/v2/test_all_eval_prompt_convert.jsonl \
    --image-folder /hy-tmp/MuirBench/v2/test_image \
    --answers-file /hy-tmp/MuirBench/v2/answer_all.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_MUIRBENCH \
    --annotation-file /hy-tmp/MuirBench/v2/test_all.json \
    --result-file /hy-tmp/MuirBench/v2/answer_all.jsonl