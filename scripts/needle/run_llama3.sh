#!/bin/bash
# Needle-in-a-Haystack on Llama-3.1-8B-Instruct (Appendix Figures 7-8)
# Tests at budget=128L and budget=1024L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct

EXPERIMENTS=(
    "full 128 LLaMA3"
    "StreamingLLM 128 LLaMA3"
    "SnapKV 128 LLaMA3"
    "restkv 128 LLaMA3"
    "full 1024 LLaMA3"
    "StreamingLLM 1024 LLaMA3"
    "SnapKV 1024 LLaMA3"
    "restkv 1024 LLaMA3"
)

CUDA_DEVICES=("0" "1" "2" "3")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r method capacity provider <<< "$exp"
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    export CUDA_VISIBLE_DEVICES=$gpu
    python -u run_needle_in_haystack.py \
        --s_len 1000 --e_len 32001 \
        --model_provider ${provider} \
        --model_name ${MODEL_PATH} \
        --attn_implementation flash_attention_2 \
        --step 200 \
        --method ${method} \
        --max_capacity_prompt ${capacity} \
        --model_version "LLaMA3_${method}_${capacity}" &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All Needle Llama3 experiments completed."
