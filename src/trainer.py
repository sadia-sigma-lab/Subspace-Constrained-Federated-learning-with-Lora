from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from src.aggregation import aggregate_fedavg, aggregate_fedsvd, aggregate_subspace_reg, aggregate_svd_like, build_reference_bundle
from src.client import train_one_client
from src.data import dirichlet_partition, iid_partition, load_raw_hellaswag, partition_statistics
from src.eval import evaluate_global_model
from src.lora_utils import extract_trainable_state_dict, load_trainable_state_dict
from src.metrics import (
    aggregate_update_norm,
    cancellation_ratio,
    mean_client_global_basis_overlap,
    mean_client_global_cosine,
    mean_pairwise_client_cosine,
    peak_round_summary,
    state_delta_norm,
)
from src.models import create_model_and_tokenizer
from src.utils import append_jsonl, dump_json, ensure_dir, safe_mean


def _select_clients(num_clients: int, fraction: float, round_idx: int, seed: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(seed + round_idx)
    num_selected = max(1, int(math.ceil(num_clients * fraction)))
    perm = torch.randperm(num_clients, generator=g).tolist()
    return sorted(perm[:num_selected])


def _save_global_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run_federated_experiment(cfg: dict[str, Any]) -> None:
    out_dir = ensure_dir(cfg['output_dir'])
    metrics_path = out_dir / 'metrics.jsonl'

    model, tokenizer, model_family = create_model_and_tokenizer(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_examples, eval_examples = load_raw_hellaswag(
        max_train_samples=cfg.get('max_train_samples'),
        max_eval_samples=cfg.get('max_eval_samples'),
        seed=int(cfg.get('seed', 42)),
    )

    if str(cfg.get('partition_type', 'dirichlet')) == 'iid':
        client_partitions = iid_partition(train_examples, num_clients=int(cfg['num_clients']), seed=int(cfg['seed']))
    else:
        client_partitions = dirichlet_partition(
            train_examples,
            num_clients=int(cfg['num_clients']),
            alpha=float(cfg.get('dirichlet_alpha', 0.3)),
            seed=int(cfg['seed']),
            field=str(cfg.get('partition_field', 'activity_label')),
            min_size=int(cfg.get('min_client_size', 1)),
        )
    dump_json(partition_statistics(client_partitions, field=str(cfg.get('partition_field', 'activity_label'))), out_dir / 'partition_stats.json')

    global_state = extract_trainable_state_dict(model)
    round_logs: list[dict[str, Any]] = []
    max_effective_lora_update_norm = cfg.get('max_effective_lora_update_norm', None)
    best_accuracy = float('-inf')
    best_round = 0
    best_state = global_state

    for round_idx in range(int(cfg['num_rounds'])):
        selected_clients = _select_clients(
            num_clients=int(cfg['num_clients']),
            fraction=float(cfg['client_fraction']),
            round_idx=round_idx,
            seed=int(cfg['seed']),
        )
        refs = build_reference_bundle(model, global_state)

        client_states = []
        client_sizes = []
        client_losses = []
        client_norms = []

        print(f"\n=== Round {round_idx + 1}/{cfg['num_rounds']} | clients={selected_clients} ===")
        for client_id in selected_clients:
            client_examples = client_partitions[client_id]
            result = train_one_client(
                model=model,
                tokenizer=tokenizer,
                model_family=model_family,
                global_trainable_state=global_state,
                client_examples=client_examples,
                cfg=cfg,
                refs=refs,
            )
            client_states.append(result.state_dict)
            client_sizes.append(result.num_examples)
            client_losses.append(result.train_loss)
            client_norms.append(state_delta_norm(model, result.state_dict))

        weights = [float(n) / max(sum(client_sizes), 1) for n in client_sizes]
        algorithm = str(cfg['algorithm'])
        if algorithm == 'fedavg':
            global_state = aggregate_fedavg(client_states, weights)
        elif algorithm == 'svd':
            global_state = aggregate_svd_like(
                model=model,
                client_states=client_states,
                weights=weights,
                rank=int(cfg['lora_rank']),
                factorization='symmetric',
                max_effective_lora_update_norm=max_effective_lora_update_norm,
            )
        elif algorithm == 'fedsvd':
            global_state = aggregate_fedsvd(
                model=model,
                current_global_state=global_state,
                client_states=client_states,
                weights=weights,
                rank=int(cfg['lora_rank']),
                max_effective_lora_update_norm=max_effective_lora_update_norm,
            )
        elif algorithm == 'subspace_reg':
            global_state = aggregate_subspace_reg(
                model=model,
                client_states=client_states,
                weights=weights,
                rank=int(cfg['lora_rank']),
                max_effective_lora_update_norm=max_effective_lora_update_norm,
            )
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')

        load_trainable_state_dict(model, global_state, device)
        eval_metrics = evaluate_global_model(
            model=model,
            tokenizer=tokenizer,
            model_family=model_family,
            global_trainable_state=global_state,
            eval_examples=eval_examples,
            cfg=cfg,
        )
        agg_norm = aggregate_update_norm(model, client_states, weights)
        cancel_ratio = cancellation_ratio(model, client_states, weights)
        client_global_cos = mean_client_global_cosine(model, client_states, weights)
        pairwise_cos = mean_pairwise_client_cosine(model, client_states)
        basis_overlap = mean_client_global_basis_overlap(global_state, client_states)

        log = {
            'round': round_idx + 1,
            'selected_clients': selected_clients,
            'mean_client_loss': safe_mean(client_losses),
            'mean_client_update_norm': safe_mean(client_norms),
            'aggregated_update_norm': agg_norm,
            'cancellation_ratio': cancel_ratio,
            'mean_client_global_cosine': client_global_cos,
            'mean_pairwise_client_cosine': pairwise_cos,
            'mean_client_global_basis_overlap': basis_overlap,
            **eval_metrics,
        }
        append_jsonl(log, metrics_path)
        round_logs.append(log)
        print(log)

        if eval_metrics['eval_accuracy'] > best_accuracy:
            best_accuracy = float(eval_metrics['eval_accuracy'])
            best_round = round_idx + 1
            best_state = {k: v.clone() for k, v in global_state.items()}
            _save_global_state(out_dir / 'best_global_state.pt', best_state)

        if (round_idx + 1) % int(cfg.get('checkpoint_every', 1)) == 0:
            _save_global_state(out_dir / f'global_state_round_{round_idx + 1:03d}.pt', global_state)

    final_metrics = round_logs[-1] if round_logs else {}
    summary = {
        'algorithm': str(cfg['algorithm']),
        'model_name': str(cfg['model_name']),
        'seed': int(cfg.get('seed', 42)),
        'num_rounds': int(cfg['num_rounds']),
        **peak_round_summary(round_logs),
        'final_eval_accuracy': float(final_metrics.get('eval_accuracy', 0.0)),
        'final_eval_loss': float(final_metrics.get('eval_loss', 0.0)),
        'final_cancellation_ratio': float(final_metrics.get('cancellation_ratio', 0.0)),
        'final_mean_client_global_cosine': float(final_metrics.get('mean_client_global_cosine', 0.0)),
        'final_mean_pairwise_client_cosine': float(final_metrics.get('mean_pairwise_client_cosine', 0.0)),
        'final_mean_client_global_basis_overlap': float(final_metrics.get('mean_client_global_basis_overlap', 0.0)),
        'best_round': int(best_round),
        'best_eval_accuracy': float(best_accuracy if best_accuracy != float('-inf') else 0.0),
    }
    dump_json(final_metrics, out_dir / 'final_metrics.json')
    dump_json(summary, out_dir / 'summary.json')
    print('\nFinished. Final metrics:')
    print(final_metrics)
    print('\nSummary:')
    print(summary)
