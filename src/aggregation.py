from __future__ import annotations

import torch

from src.lora_utils import (
    compute_delta_from_pair,
    compute_global_a_dict,
    compute_global_b_dict,
    compute_global_delta_dict,
    factorize_effective_delta,
    get_lora_pairs_from_state,
    get_scaling_factor,
    split_lora_and_other,
    weighted_average_state_dict,
)


def aggregate_fedavg(client_states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    return weighted_average_state_dict(client_states, weights)


def _average_other_params(client_states: list[dict[str, torch.Tensor]], weights: list[float], other_ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not other_ref:
        return {}
    return weighted_average_state_dict([{k: client[k] for k in other_ref.keys()} for client in client_states], weights)


def aggregate_svd_like(
    model: torch.nn.Module,
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
    rank: int,
    factorization: str = 'symmetric',
    max_effective_lora_update_norm: float | None = None,
) -> dict[str, torch.Tensor]:
    reference = client_states[0]
    lora_ref, other_ref = split_lora_and_other(reference)
    out_other = _average_other_params(client_states, weights, other_ref)
    out_lora: dict[str, torch.Tensor] = {}

    for a_key, b_key in get_lora_pairs_from_state(lora_ref):
        scaling = get_scaling_factor(model, b_key)
        delta_effective = None
        for w, client in zip(weights, client_states):
            client_delta = compute_delta_from_pair(client[a_key], client[b_key], scaling)
            delta_effective = client_delta * w if delta_effective is None else delta_effective + client_delta * w
        new_a, new_b = factorize_effective_delta(
            delta_effective=delta_effective,
            scaling=scaling,
            rank=rank,
            factorization=factorization,
            max_effective_fro_norm=max_effective_lora_update_norm,
        )
        out_lora[a_key] = new_a.cpu().to(dtype=reference[a_key].dtype)
        out_lora[b_key] = new_b.cpu().to(dtype=reference[b_key].dtype)

    out = {}
    out.update(out_lora)
    out.update(out_other)
    return out


def aggregate_fedsvd(
    model: torch.nn.Module,
    current_global_state: dict[str, torch.Tensor],
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
    rank: int,
    max_effective_lora_update_norm: float | None = None,
) -> dict[str, torch.Tensor]:
    reference = client_states[0]
    lora_ref, other_ref = split_lora_and_other(reference)
    out_other = _average_other_params(client_states, weights, other_ref)
    out_lora: dict[str, torch.Tensor] = {}

    for a_key, b_key in get_lora_pairs_from_state(lora_ref):
        avg_b = sum(w * client[b_key].float() for client, w in zip(client_states, weights))
        prev_a = current_global_state[a_key].float()
        scaling = get_scaling_factor(model, b_key)
        delta_effective = (avg_b @ prev_a) * float(scaling)
        new_a, new_b = factorize_effective_delta(
            delta_effective=delta_effective,
            scaling=scaling,
            rank=rank,
            factorization='orthogonal_a',
            max_effective_fro_norm=max_effective_lora_update_norm,
        )
        out_lora[a_key] = new_a.cpu().to(dtype=reference[a_key].dtype)
        out_lora[b_key] = new_b.cpu().to(dtype=reference[b_key].dtype)

    out = {}
    out.update(out_lora)
    out.update(out_other)
    return out


def aggregate_subspace_reg(
    model: torch.nn.Module,
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
    rank: int,
    max_effective_lora_update_norm: float | None = None,
) -> dict[str, torch.Tensor]:
    """
    Proposal-aligned server step.

    We aggregate the effective client deltas, then reparameterize each layer so
    the shared B basis becomes orthonormal (left singular vectors), matching the
    proposal's shared global reference subspace B_g.
    """
    return aggregate_svd_like(
        model=model,
        client_states=client_states,
        weights=weights,
        rank=rank,
        factorization='orthogonal_b',
        max_effective_lora_update_norm=max_effective_lora_update_norm,
    )


def build_reference_bundle(model: torch.nn.Module, global_state: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        'global_delta': compute_global_delta_dict(model, global_state),
        'global_b': compute_global_b_dict(global_state),
        'global_a': compute_global_a_dict(global_state),
    }
