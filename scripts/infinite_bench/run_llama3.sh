#!/bin/bash
# InfiniteBench on Llama-3.1-8B-Instruct (Appendix Table 14)
# Budget=1024L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_infinite_bench"
ATTN="flash_attention_2"

EXPERIMENTS=(
    "SnapKV 1024 32"
    "restkv 1024 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000 --kernel_sizes 21"
)

CUDA_DEVICES=("0" "1")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r method capacity window extra <<< "$exp"
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    export CUDA_VISIBLE_DEVICES=$gpu
    python3 run_infinite_bench.py \
        --method ${method} \
        --model_path ${MODEL_PATH} \
        --max_capacity_prompts ${capacity} \
        --attn_implementation ${ATTN} \
        --save_dir ${SAVE_DIR} \
        --use_cache True \
        --window_size ${window} \
        ${extra} &

    IDX=$((IDX + 1))
    if (( IDX % NUM_GPUS == 0 )); then
        wait
    fi
done
wait
echo "All InfiniteBench Llama3 experiments completed."
