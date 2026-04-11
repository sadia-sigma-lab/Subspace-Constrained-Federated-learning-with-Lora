from __future__ import annotations

import math
from typing import Dict

import torch


def extract_trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Export all LoRA factors plus any other trainable parameters.

    FedSVD freezes A on the client side, but the server still needs the current
    A factors to rebuild complete LoRA pairs. Keeping all LoRA tensors in the
    exchanged state also makes metric computation consistent across algorithms.
    """
    out: dict[str, torch.Tensor] = {}
    state = model.state_dict()
    for name, param in model.named_parameters():
        is_lora = ('.lora_A.' in name) or ('.lora_B.' in name)
        if is_lora or param.requires_grad:
            out[name] = state[name].detach().cpu().clone()
    return out


def load_trainable_state_dict(model: torch.nn.Module, trainable_state: dict[str, torch.Tensor], device: torch.device) -> None:
    with torch.no_grad():
        param_map = dict(model.named_parameters())
        for name, tensor in trainable_state.items():
            if name not in param_map:
                continue
            param_map[name].data.copy_(tensor.to(device=device, dtype=param_map[name].dtype))


def get_lora_pairs_from_state(trainable_state: dict[str, torch.Tensor]) -> list[tuple[str, str]]:
    keys = set(trainable_state.keys())
    pairs: list[tuple[str, str]] = []
    for key in sorted(keys):
        if '.lora_A.' in key:
            b_key = key.replace('.lora_A.', '.lora_B.')
            if b_key in keys:
                pairs.append((key, b_key))
    return pairs


def split_lora_and_other(trainable_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    lora = {}
    other = {}
    for name, tensor in trainable_state.items():
        if '.lora_A.' in name or '.lora_B.' in name:
            lora[name] = tensor
        else:
            other[name] = tensor
    return lora, other


def get_scaling_factor(model, b_key: str) -> float:
    module_name = b_key.split('.lora_B.')[0]
    module = dict(model.named_modules())[module_name]
    if hasattr(module, 'scaling'):
        scaling = module.scaling
        if isinstance(scaling, dict):
            return float(scaling['default'])
        if isinstance(scaling, (float, int)):
            return float(scaling)
    return 1.0


def compute_delta_from_pair(a_weight: torch.Tensor, b_weight: torch.Tensor, scaling: float) -> torch.Tensor:
    return (b_weight.float() @ a_weight.float()) * float(scaling)


def _svd_rank(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    r = min(rank, s.numel())
    return u[:, :r], s[:r], vh[:r, :]


def factorize_delta_symmetric(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    u_r, s_r, vh_r = _svd_rank(delta, rank)
    sqrt_s = torch.sqrt(torch.clamp(s_r, min=0.0))
    b = u_r * sqrt_s.unsqueeze(0)
    a = sqrt_s.unsqueeze(1) * vh_r
    return a, b


def factorize_delta_orthogonal_a(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A has orthonormal rows (FedSVD-style)."""
    u_r, s_r, vh_r = _svd_rank(delta, rank)
    a = vh_r
    b = u_r * s_r.unsqueeze(0)
    return a, b


def factorize_delta_orthogonal_b(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """B has orthonormal columns (proposal-style shared basis B_g)."""
    u_r, s_r, vh_r = _svd_rank(delta, rank)
    b = u_r
    a = s_r.unsqueeze(1) * vh_r
    return a, b


def maybe_clip_frobenius(delta: torch.Tensor, max_fro_norm: float | None) -> torch.Tensor:
    if max_fro_norm is None or max_fro_norm <= 0:
        return delta
    norm = float(torch.linalg.matrix_norm(delta.float(), ord='fro').item())
    if norm <= max_fro_norm:
        return delta
    return delta * (float(max_fro_norm) / max(norm, 1e-12))


def factorize_effective_delta(
    delta_effective: torch.Tensor,
    scaling: float,
    rank: int,
    factorization: str = 'symmetric',
    max_effective_fro_norm: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Factorize an *effective* LoRA update Delta = scaling * (B @ A).

    We de-scale before factorization and then reload A/B into PEFT modules that
    will apply the scaling exactly once during the forward pass.
    """
    safe_scaling = float(scaling) if abs(float(scaling)) > 1e-12 else 1.0
    target_effective = maybe_clip_frobenius(delta_effective.float(), max_effective_fro_norm)
    target_unscaled = target_effective / safe_scaling

    if factorization == 'orthogonal_a':
        new_a, new_b = factorize_delta_orthogonal_a(target_unscaled, rank=rank)
    elif factorization == 'orthogonal_b':
        new_a, new_b = factorize_delta_orthogonal_b(target_unscaled, rank=rank)
    else:
        new_a, new_b = factorize_delta_symmetric(target_unscaled, rank=rank)

    recon_effective = (new_b @ new_a) * safe_scaling
    target_norm = float(torch.linalg.matrix_norm(target_effective, ord='fro').item())
    recon_norm = float(torch.linalg.matrix_norm(recon_effective, ord='fro').item())
    if target_norm > 0.0 and recon_norm > 0.0:
        ratio = math.sqrt(target_norm / max(recon_norm, 1e-12))
        new_a = new_a * ratio
        new_b = new_b * ratio
    return new_a, new_b


def orthonormalize_columns(mat: torch.Tensor) -> torch.Tensor:
    if mat.numel() == 0:
        return mat
    q, _ = torch.linalg.qr(mat.float(), mode='reduced')
    return q


def subspace_overlap_score(left_basis: torch.Tensor, right_basis: torch.Tensor) -> float:
    """Returns normalized projector overlap in [0, 1] after QR orthonormalization."""
    if left_basis.numel() == 0 or right_basis.numel() == 0:
        return 0.0
    ql = orthonormalize_columns(left_basis)
    qr = orthonormalize_columns(right_basis)
    k = min(ql.shape[1], qr.shape[1])
    if k == 0:
        return 0.0
    overlap = torch.linalg.matrix_norm(ql.T @ qr, ord='fro').item() / math.sqrt(k)
    return float(max(0.0, min(1.0, overlap)))


def compute_global_delta_dict(model: torch.nn.Module, trainable_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    deltas: dict[str, torch.Tensor] = {}
    for a_key, b_key in get_lora_pairs_from_state(trainable_state):
        scaling = get_scaling_factor(model, b_key)
        name = b_key.split('.lora_B.')[0]
        deltas[name] = compute_delta_from_pair(trainable_state[a_key], trainable_state[b_key], scaling)
    return deltas


def compute_global_b_dict(trainable_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in trainable_state.items():
        if '.lora_B.' in name:
            out[name.split('.lora_B.')[0]] = tensor.detach().cpu().clone().float()
    return out


def compute_global_a_dict(trainable_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in trainable_state.items():
        if '.lora_A.' in name:
            out[name.split('.lora_A.')[0]] = tensor.detach().cpu().clone().float()
    return out


def freeze_lora_a_only(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if '.lora_A.' in name:
            param.requires_grad = False
        elif '.lora_B.' in name:
            param.requires_grad = True


def unfreeze_lora_a_b(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            param.requires_grad = True


def weighted_average_state_dict(client_states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    shared_keys = set(client_states[0].keys())
    for state in client_states[1:]:
        shared_keys &= set(state.keys())
    out: dict[str, torch.Tensor] = {}
    for key in sorted(shared_keys):
        out[key] = sum(w * client[key].float() for client, w in zip(client_states, weights)).cpu()
    return out
