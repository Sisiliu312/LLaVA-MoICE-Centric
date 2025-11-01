#!/bin/bash
cd /root/LLaVA
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-name /hy-tmp/checkpoints/llava-v1.5-7b-boolq \
    --question-file /hy-tmp/boolq/boolq_llava_prompt_rest_eval.jsonl \
    --answers-file /hy-tmp/boolq/answer_all.jsonl \

python -m llava.eval.eval_MUIRBENCH \
    --annotation-file /hy-tmp/boolq/boolq_llava_prompt_rest.json \
    --result-file  /hy-tmp/boolq/answer_all.jsonl

# if [ $? -eq 0 ]; then
#     echo "Training completed successfully. Shutting down..."
#     sudo shutdown now
# else
#     echo "Training failed."
#     sudo shutdown now
# fi