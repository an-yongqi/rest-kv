# [ICLR 2026] ReST-KV: 基于逐层输出重建与时空平滑的鲁棒KV缓存驱逐方法

> __ReST-KV: Robust KV Cache Eviction with Layer-wise Output Reconstruction and Spatial-Temporal Smoothing__ [[论文]](https://openreview.net/forum?id=PhEHuo7oMm)  
> _安永琪, 卢畅, 朱宽, 于涛, 赵朝阳, 吴宏, 唐明, 王金桥_  
> _中国科学院自动化研究所_

## 简介

现有的KV缓存驱逐方法仅依赖注意力权重来决定保留哪些token，忽视了驱逐操作对模型输出的实际影响。**ReST-KV** 针对这一局限性，提出了以下改进：

- **输出重建指标**：通过结合注意力优势比（odds ratio）与逐层输出重建误差来衡量每个token的重要性，同时捕获token*被关注的程度*及其对输出的*独特贡献*。
- **空间平滑**：在每个注意力头内进行自适应窗口池化，池化核大小和偏移量根据注意力分布的动态变化自适应调整。
- **时间平滑**：在观察窗口内的查询上应用指数移动平均（EMA），实现稳定的重要性聚合。
- **即插即用**：兼容现有的预算分配策略（PyramidKV, AdaKV）和KV缓存量化方法（KIVI, KVQuant）。
- **开销极低**：相比SnapKV仅增加约2%的预填充延迟，同时在解码阶段实现超过10倍的加速。

#### 支持的模型：

| 模型 | Flash Attention 2 | SDPA | Eager |
|------|:-:|:-:|:-:|
| LLaMA-2 / LLaMA-3 / LLaMA-3.1 (7B, 8B, 13B, 70B) | Y | Y | Y |
| Mistral-7B-Instruct | Y | Y | Y |
| Qwen2 / Qwen2.5 | Y | - | - |
| Gemma | Y | - | - |

## 目录

- [快速开始](#快速开始)
- [参数配置](#参数配置)
- [评测基准](#评测基准)
- [实验结果](#实验结果)
- [可视化](#可视化)
- [项目结构](#项目结构)
- [致谢](#致谢)
- [引用](#引用)

## 快速开始

### 安装

```bash
git clone https://github.com/an-yongqi/rest-kv.git
cd rest-kv
pip install -r requirements.txt
pip install -e .
```

### 数据准备

运行实验前，请将基准数据集放在 `data/` 目录下：

```
data/
├── LongBench/           # 从 https://huggingface.co/datasets/THUDM/LongBench 下载
├── RULER/               # 通过RULER的数据生成脚本生成
├── PaulGrahamEssays/    # 大海捞针实验使用
└── heads_score/         # HeadKV基线的预计算头部得分
```

对于LongBench，下载数据集并将 `.jsonl` 文件放入 `data/LongBench/`。对于RULER，按照 [RULER](https://github.com/hsiehjackson/RULER) 仓库生成评测数据并放入 `data/RULER/`。

### 最小示例

```bash
bash scripts/longbench/run.sh 0 restkv 128 <模型路径> \
    --use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000
```

在LongBench上运行ReST-KV，每层保留128个KV token。

## 参数配置

#### ReST-KV 参数

```bash
python3 run_longbench.py \
    --method restkv \
    --model_path <模型路径> \
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

参数说明：

- `--method`：驱逐方法。支持 `restkv`、`snapkv`、`h2o`、`streamingllm`、`pyramidkv`、`cam`、`l2norm`、`headkv`、`adakv`。
- `--model_path`：预训练模型路径（本地路径或Hugging Face hub ID）。
- `--max_capacity_prompts`：驱逐后每层保留的KV token数量，如 64、128、512、1024。
- `--attn_implementation`：注意力后端。支持 `flash_attention_2`、`sdpa`、`eager`。
- `--window_size`：观察窗口大小（最近的token始终保留）。默认值：`32`。
- `--use_wo`：设置后，在计算重建误差前将value通过W_o投影（输出空间指标）。
- `--use_ema`：设置后，使用EMA时间平滑替代简单均值进行重要性聚合。
- `--alpha`：EMA平滑系数。默认值：`0.3`。
- `--metric_mode`：`before` 先逐查询计算指标再聚合；`after` 先聚合注意力再计算指标。默认值：`after`。
- `--pooling`：空间平滑策略：`avgpool`、`maxpool` 或 `adaptive`。默认值：`adaptive`。
- `--scale`：自适应池化核大小的缩放因子。默认值：`2000`。
- `--tau`：value差异项的指数（仅在 `after` 模式下生效）。默认值：`1.0`。
- `--kernel_size`：池化核大小。默认值：`5`。
- `--use_pyramid`：启用PyramidKV风格的逐层自适应预算分配。
- `--merge`：可选的驱逐token合并策略，如 `pivot`（LOOK-M风格）。

## 评测基准

我们在四个基准上评测ReST-KV。实验脚本位于 `scripts/` 目录下。

#### LongBench

```bash
# 运行实验
bash scripts/longbench/run_llama3.sh

# 评测结果
bash scripts/longbench/eval.sh output_dir/results_longbench
```

#### RULER

```bash
bash scripts/ruler/run_llama3.sh
bash scripts/ruler/eval.sh output_dir/results_ruler
```

#### 大海捞针 (Needle-in-a-Haystack)

```bash
bash scripts/needle/run_mistral.sh
```

#### InfiniteBench

```bash
bash scripts/infinite_bench/run_llama3.sh
bash scripts/infinite_bench/eval.sh output_dir/results_infinite_bench
```

## 实验结果

#### LongBench

在Llama-3.1-8B-Instruct上，不同缓存预算下16个任务的平均准确率：

<p align="center">
<img src="figures/longbench_llama3.1.pdf" width="80%">
</p>

#### RULER 基准

在4K到128K上下文长度上的RULER性能（Llama-3.1-8B-Instruct，预算=1024）：

| 方法 | 4K | 8K | 16K | 32K | 64K | 128K | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full KV | 99.34 | 98.83 | 98.55 | 94.89 | 89.85 | 79.32 | 93.46 |
| StreamingLLM | 39.81 | 18.42 | 12.10 | 10.57 | 9.91 | 8.18 | 16.50 |
| SnapKV | 83.60 | 75.54 | 71.12 | 66.95 | 57.47 | 47.99 | 67.11 |
| PyramidKV | 81.35 | 73.66 | 70.23 | 69.83 | 57.84 | 48.93 | 66.97 |
| **ReST-KV** | **94.01** | **86.66** | **84.12** | **81.87** | **78.65** | **68.28** | **82.27** |

#### 效率

ReST-KV在128K上下文下实现约36%的峰值内存减少和超过10倍的解码加速，相比SnapKV仅增加约2%的预填充开销。

<p align="center">
<img src="figures/memory_mistral.pdf" width="45%">&nbsp;&nbsp;
<img src="figures/latency_mistral.pdf" width="45%">
</p>

更多结果请参阅论文。

## 可视化

可视化工具位于 `visualization/` 目录：

- `visualization/viztools/` — 注意力热力图与分析工具
- `visualization/visualization.ipynb` — 注意力模式可视化交互式notebook
- `visualization/fig_*.py` — 论文图表生成脚本
- `visualization/visualize.py` — 大海捞针结果可视化

## 项目结构

```
rest-kv/
├── restkv/                  # 核心库
│   ├── monkeypatch.py       # 运行时注意力替换
│   ├── restkv_utils.py      # ReST-KV及所有驱逐策略
│   ├── llama_model.py       # LLaMA注意力前向函数
│   ├── mistral_model.py     # Mistral注意力前向函数
│   ├── qwen_model.py        # Qwen2注意力前向函数
│   ├── gemma_model.py       # Gemma注意力前向函数
│   └── ...
├── run_longbench.py         # LongBench运行脚本
├── run_needle_in_haystack.py# 大海捞针运行脚本
├── run_ruler.py             # RULER运行脚本
├── run_infinite_bench.py    # InfiniteBench运行脚本
├── eval.py                  # LongBench评测
├── eval_ruler.py            # RULER评测
├── eval_infinite.py         # InfiniteBench评测
├── data/                    # 基准数据集（未包含，见数据准备）
├── visualization/           # 可视化工具与图表脚本
├── scripts/                 # 实验启动脚本
│   ├── longbench/           # LongBench实验与消融实验
│   ├── ruler/               # RULER基准
│   ├── needle/              # 大海捞针
│   └── infinite_bench/      # InfiniteBench
└── requirements.txt
```

## 致谢

本代码库基于 [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory)（PyramidKV）构建。衷心感谢作者提供的优秀开源框架。

## 引用

如果您觉得本项目有用，请引用

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
