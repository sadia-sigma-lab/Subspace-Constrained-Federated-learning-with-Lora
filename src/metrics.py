from __future__ import annotations

import math
import torch

from src.lora_utils import compute_delta_from_pair, get_lora_pairs_from_state, get_scaling_factor, subspace_overlap_score


def state_delta_norm(model: torch.nn.Module, trainable_state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for a_key, b_key in get_lora_pairs_from_state(trainable_state):
        scaling = get_scaling_factor(model, b_key)
        delta = compute_delta_from_pair(trainable_state[a_key], trainable_state[b_key], scaling)
        total += float((delta ** 2).sum().item())
    return total ** 0.5


def aggregate_update_norm(model: torch.nn.Module, client_states: list[dict[str, torch.Tensor]], weights: list[float]) -> float:
    total = 0.0
    reference = client_states[0]
    for a_key, b_key in get_lora_pairs_from_state(reference):
        scaling = get_scaling_factor(model, b_key)
        delta = None
        for w, client in zip(weights, client_states):
            client_delta = compute_delta_from_pair(client[a_key], client[b_key], scaling)
            delta = client_delta * w if delta is None else delta + client_delta * w
        total += float((delta ** 2).sum().item())
    return total ** 0.5


def cancellation_ratio(model: torch.nn.Module, client_states: list[dict[str, torch.Tensor]], weights: list[float]) -> float:
    client_norm_sum = 0.0
    for w, state in zip(weights, client_states):
        client_norm_sum += w * state_delta_norm(model, state)
    agg_norm = aggregate_update_norm(model, client_states, weights)
    if client_norm_sum <= 1e-12:
        return 0.0
    return agg_norm / client_norm_sum


def _flattened_effective_delta(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for a_key, b_key in get_lora_pairs_from_state(state):
        scaling = get_scaling_factor(model, b_key)
        delta = compute_delta_from_pair(state[a_key], state[b_key], scaling)
        chunks.append(delta.reshape(-1).to(dtype=torch.float64))
    if not chunks:
        return torch.zeros(1, dtype=torch.float64)
    return torch.cat(chunks)


def _safe_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    vec_a = vec_a.to(dtype=torch.float64)
    vec_b = vec_b.to(dtype=torch.float64)
    na = float(torch.linalg.vector_norm(vec_a).item())
    nb = float(torch.linalg.vector_norm(vec_b).item())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    value = float(torch.dot(vec_a, vec_b).item() / max(na * nb, 1e-12))
    return float(max(-1.0, min(1.0, value)))


def mean_client_global_cosine(model: torch.nn.Module, client_states: list[dict[str, torch.Tensor]], weights: list[float]) -> float:
    if not client_states:
        return 0.0
    weight_sum = max(sum(float(w) for w in weights), 1e-12)
    normed_weights = [float(w) / weight_sum for w in weights]
    global_vec = sum(w * _flattened_effective_delta(model, state) for w, state in zip(normed_weights, client_states))
    values = []
    for state in client_states:
        client_vec = _flattened_effective_delta(model, state)
        values.append(_safe_cosine(client_vec, global_vec))
    return float(sum(values) / len(values)) if values else 0.0


def mean_pairwise_client_cosine(model: torch.nn.Module, client_states: list[dict[str, torch.Tensor]]) -> float:
    if len(client_states) <= 1:
        return 1.0
    vecs = [_flattened_effective_delta(model, state) for state in client_states]
    values = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            values.append(_safe_cosine(vecs[i], vecs[j]))
    return float(sum(values) / len(values)) if values else 0.0


def mean_client_global_basis_overlap(global_state: dict[str, torch.Tensor], client_states: list[dict[str, torch.Tensor]]) -> float:
    if not client_states:
        return 0.0
    reference = client_states[0]
    values = []
    for a_key, b_key in get_lora_pairs_from_state(reference):
        if b_key not in global_state:
            continue
        global_b = global_state[b_key].float()
        for state in client_states:
            client_b = state[b_key].float()
            values.append(subspace_overlap_score(client_b, global_b))
    return float(sum(values) / len(values)) if values else 0.0


def peak_round_summary(round_logs: list[dict]) -> dict[str, float | int]:
    if not round_logs:
        return {'best_round': 0, 'best_eval_accuracy': 0.0, 'mean_last5_eval_accuracy': 0.0}
    best = max(round_logs, key=lambda x: float(x.get('eval_accuracy', 0.0)))
    tail = round_logs[-min(5, len(round_logs)):]
    mean_last5 = sum(float(x.get('eval_accuracy', 0.0)) for x in tail) / len(tail)
    return {
        'best_round': int(best.get('round', 0)),
        'best_eval_accuracy': float(best.get('eval_accuracy', 0.0)),
        'mean_last5_eval_accuracy': float(mean_last5),
    }
