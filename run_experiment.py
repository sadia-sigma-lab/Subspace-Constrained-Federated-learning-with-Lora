import argparse
import json
from pathlib import Path

import yaml

from src.trainer import run_federated_experiment
from src.utils import apply_overrides, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Subspace-constrained federated LoRA experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config.')
    parser.add_argument('--algorithm', type=str, default=None, choices=['fedavg', 'svd', 'fedsvd', 'subspace_reg'], help='Override algorithm from config.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional override output directory.')
    parser.add_argument('--seed', type=int, default=None, help='Optional override seed.')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override config entries as key=value. Can be used multiple times.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, args.overrides)
    if args.algorithm is not None:
        cfg['algorithm'] = args.algorithm
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed

    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / 'args.json').open('w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    set_seed(int(cfg.get('seed', 42)))
    run_federated_experiment(cfg)


if __name__ == '__main__':
    main()
