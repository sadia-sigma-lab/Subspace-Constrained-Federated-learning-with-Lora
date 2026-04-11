#!/usr/bin/env python3
"""
Run benchmark commands sequentially and save console output for each command
into its own log file, while also streaming the output live to the terminal.

Usage:
    python run_all_benchmarks.py

Optional:
    python run_all_benchmarks.py --log-dir logs
    python run_all_benchmarks.py --stop-on-error
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def sanitize_name(text: str, max_len: int = 80) -> str:
    """Create a filesystem-safe filename fragment."""
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_.-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "command"


def build_commands() -> List[str]:
    """Return the exact commands to run, in order."""
    return [
        
        
        "python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm subspace_reg --seed 43 --output_dir outputs/roberta_large_subspace_s2",
        # roberta_large seeds shown in your snippet (seed 44 only)
        "python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm fedavg --seed 44 --output_dir outputs/roberta_large_fedavg_s3",
        "python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm svd --seed 44 --output_dir outputs/roberta_large_svd_s3",
        "python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm fedsvd --seed 44 --output_dir outputs/roberta_large_fedsvd_s3",
        "python run_experiment.py --config configs/roberta_large_hellaswag.yaml --algorithm subspace_reg --seed 44 --output_dir outputs/roberta_large_subspace_s3",

        # smollm seed 42
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedavg --seed 42 --output_dir outputs/smollm_fedavg_s1",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm svd --seed 42 --output_dir outputs/smollm_svd_s1",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedsvd --seed 42 --output_dir outputs/smollm_fedsvd_s1",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm subspace_reg --seed 42 --output_dir outputs/smollm_subspace_s1",

        # smollm seed 43
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedavg --seed 43 --output_dir outputs/smollm_fedavg_s2",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm svd --seed 43 --output_dir outputs/smollm_svd_s2",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedsvd --seed 43 --output_dir outputs/smollm_fedsvd_s2",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm subspace_reg --seed 43 --output_dir outputs/smollm_subspace_s2",

        # smollm seed 44
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedavg --seed 44 --output_dir outputs/smollm_fedavg_s3",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm svd --seed 44 --output_dir outputs/smollm_svd_s3",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm fedsvd --seed 44 --output_dir outputs/smollm_fedsvd_s3",
        "python run_experiment.py --config configs/smollm_360m_hellaswag.yaml --algorithm subspace_reg --seed 44 --output_dir outputs/smollm_subspace_s3",

        # summary
        "python scripts/summarize_benchmark.py --input_root ./outputs --output_dir ./outputs/benchmark_summary",
        "find outputs/benchmark_summary -maxdepth 2 | sort",
        "cat outputs/benchmark_summary/summary.json",

        # subspace sweep
        "python scripts/run_subspace_sweep.py --project_root . --config configs/roberta_large_hellaswag.yaml --seeds 42 43 44 --mu_values 0.01 0.05 0.1 0.2 --alpha_values 0.0 0.01 0.05 --lambda_values 1e-5 --output_root ./outputs/subspace_sweep",
        "find outputs/subspace_sweep -maxdepth 2 | sort",

        # final inline Python block
        r'''python - <<'PY'
import json, os
paths = [
    "outputs/roberta_large_fedavg_s1/summary.json",
    "outputs/roberta_large_svd_s1/summary.json",
    "outputs/roberta_large_fedsvd_s1/summary.json",
    "outputs/roberta_large_subspace_s1/summary.json",
]
for p in paths:
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        print(p)
        print({
            "algorithm": d["algorithm"],
            "best_eval_accuracy": d["best_eval_accuracy"],
            "mean_last5_eval_accuracy": d["mean_last5_eval_accuracy"],
            "final_eval_accuracy": d["final_eval_accuracy"],
            "final_eval_loss": d["final_eval_loss"],
        })
        print()
PY''',
    ]


def stream_command(command: str, log_path: Path, cwd: Path) -> int:
    """Run one shell command, stream output to terminal and log file."""
    header = f"\n{'=' * 100}\nRUNNING: {command}\nLOG: {log_path}\n{'=' * 100}\n"
    print(header, flush=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(header)
        log_file.flush()

        process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        footer = f"\n[exit_code={return_code}]\n"
        print(footer, flush=True)
        log_file.write(footer)
        log_file.flush()

    return return_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="command_logs", help="Directory to store per-command log files")
    parser.add_argument("--project-root", default=".", help="Project root where commands should be executed")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if a command fails")
    args = parser.parse_args()

    cwd = Path(args.project_root).resolve()
    log_dir = cwd / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    commands = build_commands()
    summary_path = log_dir / "run_summary.txt"

    failures = 0
    start_all = time.time()

    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write(f"Project root: {cwd}\n")
        summary.write(f"Log dir: {log_dir}\n")
        summary.write(f"Total commands: {len(commands)}\n\n")

        for idx, command in enumerate(commands, start=1):
            short_name = sanitize_name(command.splitlines()[0])
            log_path = log_dir / f"{idx:02d}_{short_name}.log"

            start_cmd = time.time()
            return_code = stream_command(command, log_path, cwd)
            elapsed = time.time() - start_cmd

            summary.write(
                f"[{idx:02d}] return_code={return_code} elapsed={elapsed:.2f}s log={log_path.name}\n"
            )
            summary.flush()

            if return_code != 0:
                failures += 1
                if args.stop_on_error:
                    print(f"Stopping early because command {idx} failed.", flush=True)
                    break

        total_elapsed = time.time() - start_all
        summary.write(f"\nFinished in {total_elapsed:.2f}s\n")
        summary.write(f"Failures: {failures}\n")

    print("\nDone.")
    print(f"Per-command logs saved in: {log_dir}")
    print(f"Run summary saved in: {summary_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
