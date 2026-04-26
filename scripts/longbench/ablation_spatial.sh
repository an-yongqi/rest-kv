#!/bin/bash
# Ablation study: adaptive spatial smoothing scale factor beta (Figure 5 right)
# Model: Llama-3.1-8B-Instruct, budget=128L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_longbench_ablation"
ATTN="flash_attention_2"

SCALE_VALUES=(200 400 800 1200 1600 2000)
CUDA_DEVICES=("0" "1" "2" "3" "4" "5")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

for scale in "${SCALE_VALUES[@]}"; do
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    export CUDA_VISIBLE_DEVICES=$gpu
    python3 run_longbench.py \
        --method restkv \
        --model_path ${MODEL_PATH} \
        --max_capacity_prompts 128 \
        --attn_implementation ${ATTN} \
        --save_dir ${SAVE_DIR} \
        --use_cache True \
        --window_size 32 \
        --use_wo --use_ema \
        --alpha 0.3 \
        --metric_mode after \
        --pooling adaptive --scale ${scale} \
        --add_file_name "scale_${scale}" &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All spatial smoothing ablation experiments completed."
