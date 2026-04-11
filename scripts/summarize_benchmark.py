from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize federated LoRA benchmark runs.')
    parser.add_argument('--input_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()


def load_rows(input_root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(input_root.rglob('summary.json')):
        with summary_path.open('r', encoding='utf-8') as f:
            row = json.load(f)
        rel = summary_path.relative_to(input_root)
        row['run_dir'] = str(summary_path.parent)
        row['relative_path'] = str(rel)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_root)
    if not rows:
        raise SystemExit('No summary.json files found.')

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'all_runs.csv', index=False)

    group_cols = ['model_name', 'algorithm']
    summary = df.groupby(group_cols).agg(
        seeds=('seed', 'count'),
        mean_best_eval_accuracy=('best_eval_accuracy', 'mean'),
        std_best_eval_accuracy=('best_eval_accuracy', 'std'),
        mean_final_eval_accuracy=('final_eval_accuracy', 'mean'),
        std_final_eval_accuracy=('final_eval_accuracy', 'std'),
        mean_mean_last5_eval_accuracy=('mean_last5_eval_accuracy', 'mean'),
        mean_final_cancellation_ratio=('final_cancellation_ratio', 'mean'),
        mean_final_mean_client_global_cosine=('final_mean_client_global_cosine', 'mean'),
        mean_final_mean_client_global_basis_overlap=('final_mean_client_global_basis_overlap', 'mean'),
    ).reset_index()
    summary.to_csv(output_dir / 'benchmark_summary.csv', index=False)

    for model_name, model_df in df.groupby('model_name'):
        plt.figure(figsize=(8, 5))
        for algorithm, algo_df in model_df.groupby('algorithm'):
            plt.scatter(algo_df['seed'], algo_df['final_eval_accuracy'], label=algorithm)
        plt.title(f'Final accuracy by seed: {model_name}')
        plt.xlabel('Seed')
        plt.ylabel('Final eval accuracy')
        plt.legend()
        safe_name = model_name.replace('/', '_')
        plt.tight_layout()
        plt.savefig(output_dir / f'{safe_name}_final_accuracy_by_seed.png', dpi=150)
        plt.close()

    print('Wrote:')
    print(output_dir / 'all_runs.csv')
    print(output_dir / 'benchmark_summary.csv')


if __name__ == '__main__':
    main()
