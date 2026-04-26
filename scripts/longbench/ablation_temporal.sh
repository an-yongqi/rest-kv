#!/bin/bash
# Ablation study: EMA temporal smoothing factor alpha (Figure 5 left)
# Model: Llama-3.1-8B-Instruct, budget=128L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_longbench_ablation"
ATTN="flash_attention_2"

ALPHA_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
CUDA_DEVICES=("0" "1" "2" "3" "4" "5" "6" "7")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

for alpha in "${ALPHA_VALUES[@]}"; do
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
        --alpha ${alpha} \
        --metric_mode after \
        --pooling adaptive --scale 2000 \
        --add_file_name "alpha_${alpha}" &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All temporal smoothing ablation experiments completed."
