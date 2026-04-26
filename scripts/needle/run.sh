#!/bin/bash
# Generic Needle-in-a-Haystack runner
# Usage: bash run.sh <gpu_id> <method> <max_capacity> <model_provider> <model_path> [extra_args]
#
# model_provider: LLaMA3 or Mistral

export CUDA_VISIBLE_DEVICES=$1
method=$2
max_capacity_prompt=$3
model_provider=$4
model_path=$5
shift 5

python -u run_needle_in_haystack.py \
    --s_len 1000 --e_len 32001 \
    --model_provider ${model_provider} \
    --model_name ${model_path} \
    --attn_implementation flash_attention_2 \
    --step 200 \
    --method ${method} \
    --max_capacity_prompt ${max_capacity_prompt} \
    --model_version "${model_provider}_${method}_${max_capacity_prompt}" \
    "$@"
