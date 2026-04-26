# [ICLR 2026] ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing

> __ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing__ [[Paper]](https://openreview.net/forum?id=PhEHuo7oMm)  
> _Yongqi An, Chang Lu, Kuan Zhu, Tao Yu, Chaoyang Zhao, Hong Wu, Ming Tang, Jinqiao Wang_  
> _Institute of Automation, Chinese Academy of Sciences_

## Introduction

Existing KV cache eviction methods rely solely on attention weights to decide which tokens to keep, ignoring the actual impact of eviction on the model's output. **ReST-KV** addresses this limitation with:

- **Output-Reconstruction Indicator**: Measures each token's importance by combining its attention odds ratio with the layer-wise output reconstruction error, capturing both *how much attention* a token receives and *how uniquely it contributes* to the output.
- **Spatial Smoothing**: Adaptive window-based pooling per attention head, with kernel size and shift adjusted based on attention distribution dynamics.
- **Temporal Smoothing**: Exponential moving average (EMA) across observation window queries for stable importance aggregation.
- **Plug-and-Play**: Compatible with existing budget allocation strategies (PyramidKV, AdaKV) and KV cache quantization (KIVI, KVQuant).
- **Negligible Overhead**: Only ~2% additional prefill latency over SnapKV, with >10x decoding speedup over full cache.

#### Supported LLMs:

| Model | Flash Attention 2 | SDPA | Eager |
|-------|:-:|:-:|:-:|
| LLaMA-2 / LLaMA-3 / LLaMA-3.1 (7B, 8B, 13B, 70B) | Y | Y | Y |
| Mistral-7B-Instruct | Y | Y | Y |
| Qwen2 / Qwen2.5 | Y | - | - |
| Gemma | Y | - | - |

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Evaluation Benchmarks](#evaluation-benchmarks)
- [Results](#results)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Quick Start

### Installation

```bash
git clone https://github.com/an-yongqi/rest-kv.git
cd rest-kv
pip install -r requirements.txt
pip install -e .
```

### Data Preparation

Benchmark datasets should be placed under the `data/` directory before running experiments:

```
data/
├── LongBench/           # Download from https://huggingface.co/datasets/THUDM/LongBench
├── RULER/               # Generated via RULER's data generation scripts
├── PaulGrahamEssays/    # Used by Needle-in-a-Haystack
└── heads_score/         # Pre-computed head scores for HeadKV baseline
```

For LongBench, download the dataset and place the `.jsonl` files in `data/LongBench/`. For RULER, generate the evaluation data following the [RULER](https://github.com/hsiehjackson/RULER) repository and place them under `data/RULER/`.

### Minimal Example

```bash
bash scripts/longbench/run.sh 0 restkv 128 <path_to_model> \
    --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000
```

This runs ReST-KV on LongBench with a cache budget of 128 tokens per layer.

## Configuration

#### ReST-KV Parameters

```bash
python3 run_longbench.py \
    --method restkv \
    --model_path <path_to_model> \
    --max_capacity_prompts 128 \
    --attn_implementation flash_attention_2 \
    --save_dir output_dir/results_longbench \
    --use_cache True \
    --window_size 32 \
    --use_wo \
    --use_ema \
    --alpha 0.3 \
    --metric_mode after \
    --pooling adaptive \
    --scale 2000
```

Arguments:

- `--method`: Eviction method to use. Supports `restkv`, `snapkv`, `h2o`, `streamingllm`, `pyramidkv`, `cam`, `l2norm`, `headkv`, `adakv`.
- `--model_path`: Path to the pretrained model (local or Hugging Face hub ID).
- `--max_capacity_prompts`: Number of KV tokens retained per layer after eviction. E.g., 64, 128, 512, 1024.
- `--attn_implementation`: Attention backend. Supports `flash_attention_2`, `sdpa`, `eager`.
- `--window_size`: Size of the observation window (recent tokens always kept). Default: `32`.
- `--use_wo`: When set, projects values through W_o before computing reconstruction error (output-space metric).
- `--use_ema`: When set, uses EMA temporal smoothing instead of simple mean for importance aggregation.
- `--alpha`: EMA smoothing coefficient. Default: `0.3`.
- `--metric_mode`: `before` computes per-query metric then aggregates; `after` aggregates attention first then computes metric. Default: `after`.
- `--pooling`: Spatial smoothing strategy: `avgpool`, `maxpool`, or `adaptive`. Default: `adaptive`.
- `--scale`: Scaling factor for adaptive pooling kernel size. Default: `2000`.
- `--tau`: Exponent for value difference term (only in `after` mode). Default: `1.0`.
- `--kernel_size`: Kernel size for pooling. Default: `5`.
- `--use_pyramid`: Enable PyramidKV-style layer-adaptive budget allocation.
- `--merge`: Optional merge strategy for evicted tokens, e.g., `pivot` (LOOK-M style).

## Evaluation Benchmarks

We evaluate ReST-KV on four benchmarks. Experiment scripts are provided under `scripts/`.

#### LongBench

```bash
# Run experiments
bash scripts/longbench/run_llama3.sh

# Evaluate results
bash scripts/longbench/eval.sh output_dir/results_longbench
```

#### RULER

```bash
bash scripts/ruler/run_llama3.sh
bash scripts/ruler/eval.sh output_dir/results_ruler
```

#### Needle-in-a-Haystack

```bash
bash scripts/needle/run_mistral.sh
```

#### InfiniteBench

```bash
bash scripts/infinite_bench/run_llama3.sh
bash scripts/infinite_bench/eval.sh output_dir/results_infinite_bench
```

## Results

#### LongBench

Average accuracy across 16 tasks under varying cache budgets on Llama-3.1-8B-Instruct:

<p align="center">
<img src="figures/longbench_llama3.1.pdf" width="80%">
</p>

#### RULER Benchmark

Performance on RULER across context lengths from 4K to 128K (Llama-3.1-8B-Instruct, budget=1024L):

| Method | 4K | 8K | 16K | 32K | 64K | 128K | Avg. |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full KV | 99.34 | 98.83 | 98.55 | 94.89 | 89.85 | 79.32 | 93.46 |
| StreamingLLM | 39.81 | 18.42 | 12.10 | 10.57 | 9.91 | 8.18 | 16.50 |
| SnapKV | 83.60 | 75.54 | 71.12 | 66.95 | 57.47 | 47.99 | 67.11 |
| PyramidKV | 81.35 | 73.66 | 70.23 | 69.83 | 57.84 | 48.93 | 66.97 |
| **ReST-KV** | **94.01** | **86.66** | **84.12** | **81.87** | **78.65** | **68.28** | **82.27** |

#### Efficiency

ReST-KV achieves ~36% peak memory reduction and >10x decoding speedup at 128K context, with only ~2% prefill overhead over SnapKV.

<p align="center">
<img src="figures/memory_mistral.pdf" width="45%">&nbsp;&nbsp;
<img src="figures/latency_mistral.pdf" width="45%">
</p>

More results can be found in the paper.

## Visualization

Visualization tools are provided in the `visualization/` directory:

- `visualization/viztools/` — Attention heatmap and analysis tools
- `visualization/visualization.ipynb` — Interactive notebook for attention pattern visualization
- `visualization/fig_*.py` — Scripts for generating paper figures
- `visualization/visualize.py` — Needle-in-a-Haystack result visualization

## Project Structure

```
rest-kv/
├── restkv/                  # Core library
│   ├── monkeypatch.py       # Runtime attention replacement
│   ├── restkv_utils.py      # ReST-KV and all eviction strategies
│   ├── llama_model.py       # LLaMA attention forward functions
│   ├── mistral_model.py     # Mistral attention forward functions
│   ├── qwen_model.py        # Qwen2 attention forward functions
│   ├── gemma_model.py       # Gemma attention forward functions
│   └── ...
├── run_longbench.py         # LongBench runner
├── run_needle_in_haystack.py# Needle-in-a-Haystack runner
├── run_ruler.py             # RULER runner
├── run_infinite_bench.py    # InfiniteBench runner
├── eval.py                  # LongBench evaluation
├── eval_ruler.py            # RULER evaluation
├── eval_infinite.py         # InfiniteBench evaluation
├── data/                    # Benchmark datasets (not included, see Data Preparation)
├── visualization/           # Visualization tools and figure scripts
├── scripts/                 # Experiment launch scripts
│   ├── longbench/           # LongBench experiments & ablations
│   ├── ruler/               # RULER benchmark
│   ├── needle/              # Needle-in-a-Haystack
│   └── infinite_bench/      # InfiniteBench
└── requirements.txt
```

## Acknowledgement

This codebase is built upon [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) (PyramidKV). We sincerely thank the authors for their excellent open-source framework.

## Citation

If you find this project useful, please cite

```bibtex
@inproceedings{
an2026restkv,
title={Re{ST}-{KV}: Robust {KV} Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing},
author={Yongqi An and Chang Lu and Kuan Zhu and Tao Yu and Chaoyang Zhao and Hong Wu and Ming Tang and Jinqiao Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=PhEHuo7oMm}
}
```
