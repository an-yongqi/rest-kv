#!/bin/bash
# Needle-in-a-Haystack on Mistral-7B-Instruct-v0.3 (Figure 3, Appendix Figure 6)
# Tests at budget=128L and budget=1024L

MODEL_PATH="<path_to_model>"  # e.g., mistralai/Mistral-7B-Instruct-v0.3

EXPERIMENTS=(
    # method capacity model_provider
    "full 128 Mistral"
    "StreamingLLM 128 Mistral"
    "SnapKV 128 Mistral"
    "restkv 128 Mistral"
    "full 1024 Mistral"
    "StreamingLLM 1024 Mistral"
    "SnapKV 1024 Mistral"
    "restkv 1024 Mistral"
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
        --model_version "Mistral_${method}_${capacity}" &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All Needle Mistral experiments completed."
