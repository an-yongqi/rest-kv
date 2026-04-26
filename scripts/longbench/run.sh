#!/bin/bash
# Generic LongBench evaluation runner
# Usage: bash run.sh <gpu_id> <method> <max_capacity> <model_path> [extra_args]
#
# Example:
#   bash run.sh 0 restkv 128 meta-llama/Llama-3.1-8B-Instruct --use_wo --use_ema

export CUDA_VISIBLE_DEVICES=$1
method=$2
max_capacity_prompts=$3
model_path=$4
shift 4

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation flash_attention_2 \
    --save_dir output_dir/results_longbench \
    --use_cache True \
    "$@"
