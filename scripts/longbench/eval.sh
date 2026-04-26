#!/bin/bash
# Evaluate LongBench results
# Usage: bash eval.sh <results_dir>

results_dir=$1
python3 eval.py --results_dir ${results_dir}
