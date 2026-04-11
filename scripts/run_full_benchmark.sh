#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/run_benchmark_matrix.py   --project_root .   --configs configs/roberta_large_hellaswag.yaml configs/smollm_360m_hellaswag.yaml   --algorithms fedavg svd fedsvd subspace_reg   --seeds 42 43 44   --output_root ./outputs/benchmark_runs

python scripts/summarize_benchmark.py   --input_root ./outputs/benchmark_runs   --output_dir ./outputs/benchmark_summary
