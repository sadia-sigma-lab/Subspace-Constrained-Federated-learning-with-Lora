from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run benchmark matrix across models, algorithms, and seeds.')
    parser.add_argument('--project_root', type=str, default='.', help='Repository root containing run_experiment.py')
    parser.add_argument('--configs', nargs='+', required=True, help='Config files to run')
    parser.add_argument('--algorithms', nargs='+', default=['fedavg', 'svd', 'fedsvd', 'subspace_reg'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--output_root', type=str, default='./outputs/benchmark_runs')
    parser.add_argument('--extra_set', nargs='*', default=[], help='Extra key=value overrides applied to every run')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for config, algorithm, seed in itertools.product(args.configs, args.algorithms, args.seeds):
        config_path = Path(config)
        config_stem = config_path.stem
        out_dir = output_root / config_stem / algorithm / f'seed_{seed}'
        cmd = [
            sys.executable,
            str(root / 'run_experiment.py'),
            '--config', str(root / config_path),
            '--algorithm', algorithm,
            '--seed', str(seed),
            '--output_dir', str(out_dir),
        ]
        for item in args.extra_set:
            cmd.extend(['--set', item])
        print('\n[RUN]', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
