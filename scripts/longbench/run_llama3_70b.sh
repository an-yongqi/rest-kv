#!/bin/bash
# LongBench experiments on Llama-3-70B-Instruct (multi-GPU)
# Reproduces Appendix Table 9

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Meta-Llama-3-70B-Instruct
SAVE_DIR="output_dir/results_longbench"
ATTN="flash_attention_2"

# 70B model requires multiple GPUs per run
EXPERIMENTS=(
    "FullKV 128 32"
    "StreamingLLM 64 32"
    "StreamingLLM 512 32"
    "SnapKV 64 32"
    "SnapKV 512 32"
    "restkv 64 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
    "restkv 512 32 --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000"
)

# Use 4 GPUs per experiment for 70B model
export CUDA_VISIBLE_DEVICES=0,1,2,3

for exp in "${EXPERIMENTS[@]}"; do
    read -r method capacity window extra <<< "$exp"

    python3 run_longbench.py \
        --method ${method} \
        --model_path ${MODEL_PATH} \
        --max_capacity_prompts ${capacity} \
        --attn_implementation ${ATTN} \
        --save_dir ${SAVE_DIR} \
        --use_cache True \
        --window_size ${window} \
        ${extra}
done
echo "All LongBench Llama3-70B experiments completed."
