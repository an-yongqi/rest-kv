#!/bin/bash
# Ablation study: eviction indicator components (Table 3, Appendix Table 12)
# Tests: random, attention-only, attention+value-norm, ReST-KV (output reconstruction)
# Model: Llama-3.1-8B-Instruct, budget=128L

MODEL_PATH="<path_to_model>"  # e.g., meta-llama/Llama-3.1-8B-Instruct
SAVE_DIR="output_dir/results_longbench_ablation"
ATTN="flash_attention_2"
CAPACITY=128

# indicator: random / attn / attn-v / reconstruction
# ts (temporal smoothing): none / mean / ema / inv-ema
# ss (spatial smoothing): none / avgpool / maxpool / adaptive
EXPERIMENTS=(
    # Indicator ablation (Table 12 left)
    "Ablation ${CAPACITY} 32 --indicator random --ts none --ss none"
    "Ablation ${CAPACITY} 32 --indicator attn --ts none --ss none"
    "Ablation ${CAPACITY} 32 --indicator attn-v --ts none --ss none"
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts none --ss none"
    # Temporal smoothing ablation (Table 12 right-left)
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts mean --ss none"
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts inv-ema --ss none"
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts ema --ss none"
    # Spatial smoothing ablation (Table 12 right-right)
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts ema --ss avgpool"
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts ema --ss maxpool"
    "Ablation ${CAPACITY} 32 --indicator reconstruction --ts ema --ss adaptive"
)

CUDA_DEVICES=("0" "1" "2" "3" "4" "5")
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
echo "All indicator ablation experiments completed."
