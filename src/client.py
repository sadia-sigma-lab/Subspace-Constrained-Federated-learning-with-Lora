from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data import build_dataset_and_collator, make_dataloader
from src.lora_utils import extract_trainable_state_dict, freeze_lora_a_only, get_lora_pairs_from_state, load_trainable_state_dict, unfreeze_lora_a_b


@dataclass
class ClientResult:
    state_dict: dict[str, torch.Tensor]
    train_loss: float
    num_examples: int


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _causal_multiple_choice_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, num_choices, seqlen = batch['input_ids'].shape
    flat_input_ids = batch['input_ids'].view(bsz * num_choices, seqlen)
    flat_attention_mask = batch['attention_mask'].view(bsz * num_choices, seqlen)
    flat_loss_mask = batch['loss_mask'].view(bsz * num_choices, seqlen)

    outputs = model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = flat_input_ids[:, 1:]
    loss_mask = flat_loss_mask[:, 1:]

    token_log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(token_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * loss_mask

    lengths = loss_mask.sum(dim=-1).clamp(min=1.0)
    choice_scores = gathered.sum(dim=-1) / lengths
    choice_scores = choice_scores.view(bsz, num_choices)
    loss = F.cross_entropy(choice_scores, batch['labels'])
    return loss, choice_scores


def _regularization_term(model: torch.nn.Module, refs: dict[str, dict[str, torch.Tensor]] | None, algorithm: str, mu: float, alpha: float, lambda_reg: float, device: torch.device) -> torch.Tensor:
    if algorithm != 'subspace_reg':
        return torch.zeros((), device=device)
    refs = refs or {}
    global_delta = refs.get('global_delta', {})
    global_b = refs.get('global_b', {})
    total = torch.zeros((), device=device)
    pair_count = 0
    param_map = dict(model.named_parameters())

    for a_key, b_key in get_lora_pairs_from_state(extract_trainable_state_dict(model)):
        if a_key not in param_map or b_key not in param_map:
            continue
        a = param_map[a_key]
        b = param_map[b_key]
        layer_name = b_key.split('.lora_B.')[0]
        module = dict(model.named_modules())[layer_name]
        scaling = 1.0
        if hasattr(module, 'scaling'):
            scaling_obj = module.scaling
            if isinstance(scaling_obj, dict):
                scaling = float(scaling_obj['default'])
            else:
                scaling = float(scaling_obj)
        delta = (b.float() @ a.float()) * scaling
        layer_reg = torch.zeros((), device=device)
        if layer_name in global_delta:
            ref_delta = global_delta[layer_name].to(device=device, dtype=delta.dtype)
            layer_reg = layer_reg + 0.5 * mu * torch.sum((delta - ref_delta) ** 2)
        if layer_name in global_b:
            ref_b = global_b[layer_name].to(device=device, dtype=b.dtype)
            layer_reg = layer_reg + 0.5 * alpha * torch.sum((b - ref_b) ** 2)
        layer_reg = layer_reg + 0.5 * lambda_reg * (torch.sum(a.float() ** 2) + torch.sum(b.float() ** 2))
        total = total + layer_reg
        pair_count += 1

    if pair_count == 0:
        return torch.zeros((), device=device)
    return total / pair_count


def train_one_client(model: torch.nn.Module, tokenizer, model_family: str, global_trainable_state: dict[str, torch.Tensor], client_examples: list[dict[str, Any]], cfg: dict[str, Any], refs: dict[str, dict[str, torch.Tensor]] | None = None) -> ClientResult:
    device = next(model.parameters()).device
    algorithm = str(cfg['algorithm'])
    load_trainable_state_dict(model, global_trainable_state, device)

    if algorithm == 'fedsvd':
        freeze_lora_a_only(model)
    else:
        unfreeze_lora_a_b(model)

    dataset, collator = build_dataset_and_collator(model_family=model_family, examples=client_examples, tokenizer=tokenizer, max_length=int(cfg['max_length']))
    loader = make_dataloader(dataset=dataset, collator=collator, batch_size=int(cfg['per_device_train_batch_size']), shuffle=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(cfg['learning_rate']), weight_decay=float(cfg.get('weight_decay', 0.0)))
    grad_accum = int(cfg.get('gradient_accumulation_steps', 1))
    max_steps = int(cfg.get('max_local_steps', 0))
    num_epochs = int(cfg.get('local_epochs', 1))

    total_loss = 0.0
    seen_steps = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(num_epochs):
        progress = tqdm(loader, desc='local train', leave=False)
        for step_idx, batch in enumerate(progress):
            batch = _move_batch_to_device(batch, device)
            if model_family == 'encoder_mc':
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                task_loss = outputs.loss
            else:
                task_loss, _ = _causal_multiple_choice_loss(model, batch)

            reg = _regularization_term(
                model=model,
                refs=refs,
                algorithm=algorithm,
                mu=float(cfg.get('mu', 0.0)),
                alpha=float(cfg.get('alpha', 0.0)),
                lambda_reg=float(cfg.get('lambda_reg', 0.0)),
                device=device,
            )
            loss = (task_loss + reg) / grad_accum
            loss.backward()

            if (step_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('max_grad_norm', 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(task_loss.item())
            seen_steps += 1
            progress.set_postfix(loss=f"{task_loss.item():.4f}")

            if max_steps > 0 and seen_steps >= max_steps:
                break
        if max_steps > 0 and seen_steps >= max_steps:
            break

    if seen_steps > 0 and seen_steps % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('max_grad_norm', 1.0)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    client_state = extract_trainable_state_dict(model)
    return ClientResult(state_dict=client_state, train_loss=(total_loss / max(seen_steps, 1)), num_examples=len(client_examples))
