#!/bin/bash
# LongBench experiments on Llama-3.1-8B-Instruct
# Reproduces main results (Table 2) and Appendix Table 5

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_longbench"
ATTN="flash_attention_2"

# Experiment settings: "method max_capacity window_size extra_args"
EXPERIMENTS=(
    "FullKV 128 32"
    "StreamingLLM 64 32"
    "StreamingLLM 128 32"
    "StreamingLLM 512 32"
    "SnapKV 64 32"
    "SnapKV 128 32"
    "SnapKV 512 32"
    "H2O 64 32"
    "H2O 128 32"
    "H2O 512 32"
    "restkv 64 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
    "restkv 128 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
    "restkv 512 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
    "restkv 1024 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
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
echo "All LongBench Llama3 experiments completed."
