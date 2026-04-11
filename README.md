# Subspace-Constrained Federated LoRA: Research Benchmark Package

This repository contains the code for all experiments of Subspace-Regularized LoRA :

## Repository layout

- `run_experiment.py` — single experiment entrypoint
- `configs/` — base configs for RoBERTa-large, SmolLM-360M, smoke test
- `scripts/run_benchmark_matrix.py` — matrix runner across configs / algorithms / seeds
- `scripts/run_subspace_sweep.py` — sweep `mu`, `alpha`, `lambda_reg`
- `scripts/summarize_benchmark.py` — aggregate results and save plots
- `scripts/run_full_benchmark.sh` — one-shot benchmark shell wrapper

## Install

```bash
pip install -r requirements.txt
```

## Quick smoke test

```bash
python run_experiment.py \
  --config configs/smoke_roberta_base.yaml \
  --algorithm subspace_reg \
  --output_dir outputs/smoke_subspace
```

## Single full run

```bash
python run_experiment.py \
  --config configs/roberta_large_hellaswag.yaml \
  --algorithm subspace_reg \
  --seed 42 \
  --output_dir outputs/roberta_large_subspace_seed42
```

## Run all algorithms for RoBERTa-large

```bash
python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm fedavg       --seed 42 --output_dir outputs/roberta_large/fedavg/seed_42
python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm svd          --seed 42 --output_dir outputs/roberta_large/svd/seed_42
python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm fedsvd       --seed 42 --output_dir outputs/roberta_large/fedsvd/seed_42
python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm subspace_reg --seed 42 --output_dir outputs/roberta_large/subspace_reg/seed_42
```

## Run the full benchmark matrix (recommended)

```bash
python scripts/run_benchmark_matrix.py \
  --project_root . \
  --configs configs/roberta_large_hellaswag.yaml configs/smollm_360m_hellaswag.yaml \
  --algorithms fedavg svd fedsvd subspace_reg \
  --seeds 42 43 44 \
  --output_root ./outputs/benchmark_runs

python scripts/summarize_benchmark.py \
  --input_root ./outputs/benchmark_runs \
  --output_dir ./outputs/benchmark_summary
```

## Run a subspace sweep

```bash
python scripts/run_subspace_sweep.py \
  --project_root . \
  --config configs/roberta_large_hellaswag.yaml \
  --seeds 42 43 44 \
  --mu_values 0.01 0.05 0.1 0.2 \
  --alpha_values 0.0 0.01 0.05 \
  --lambda_values 1e-5 \
  --output_root ./outputs/subspace_sweep
```

## Config overrides without editing YAML

```bash
python run_experiment.py \
  --config configs/roberta_large_hellaswag.yaml \
  --algorithm subspace_reg \
  --seed 43 \
  --set mu=0.1 \
  --set alpha=0.02 \
  --set learning_rate=1e-4 \
  --output_dir outputs/test_override
```




