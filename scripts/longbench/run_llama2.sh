#!/bin/bash
# LongBench experiments on Llama-2-7B-Chat
# Reproduces Appendix Table 7

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-2-7b-chat-hf
SAVE_DIR="output_dir/results_longbench"
ATTN="flash_attention_2"

EXPERIMENTS=(
    "FullKV 128 32"
    "StreamingLLM 64 32"
    "StreamingLLM 512 32"
    "SnapKV 64 32"
    "SnapKV 512 32"
    "H2O 64 32"
    "H2O 512 32"
    "restkv 64 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
    "restkv 512 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
)

CUDA_DEVICES=("0" "1" "2" "3")
NUM_GPUS=${#CUDA_DEVICES[@]}
IDX=0

for exp in "${EXPERIMENTS[@]}"; do
    read -r method capacity window extra <<< "$exp"
    gpu=${CUDA_DEVICES[$((IDX % NUM_GPUS))]}

    export CUDA_VISIBLE_DEVICES=$gpu
    python3 run_longbench.py \
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
echo "All LongBench Llama2 experiments completed."
