#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
python run_experiment.py --config configs/smoke_roberta_base.yaml --algorithm fedavg
python run_experiment.py --config configs/smoke_roberta_base.yaml --algorithm svd
python run_experiment.py --config configs/smoke_roberta_base.yaml --algorithm fedsvd
python run_experiment.py --config configs/smoke_roberta_base.yaml --algorithm subspace_reg
