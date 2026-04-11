from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sweep subspace regularization hyperparameters.')
    parser.add_argument('--project_root', type=str, default='.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--mu_values', nargs='+', type=float, default=[0.01, 0.05, 0.1, 0.2])
    parser.add_argument('--alpha_values', nargs='+', type=float, default=[0.0, 0.01, 0.05])
    parser.add_argument('--lambda_values', nargs='+', type=float, default=[1e-5])
    parser.add_argument('--output_root', type=str, default='./outputs/subspace_sweep')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    for seed, mu, alpha, lam in itertools.product(args.seeds, args.mu_values, args.alpha_values, args.lambda_values):
        out_dir = output_root / config_path.stem / f'seed_{seed}' / f'mu_{mu}_alpha_{alpha}_lambda_{lam}'
        cmd = [
            sys.executable,
            str(root / 'run_experiment.py'),
            '--config', str(root / config_path),
            '--algorithm', 'subspace_reg',
            '--seed', str(seed),
            '--output_dir', str(out_dir),
            '--set', f'mu={mu}',
            '--set', f'alpha={alpha}',
            '--set', f'lambda_reg={lam}',
        ]
        print('\n[RUN]', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
