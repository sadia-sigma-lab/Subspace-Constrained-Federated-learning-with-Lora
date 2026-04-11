from __future__ import annotations

import torch
from tqdm.auto import tqdm

from src.client import _causal_multiple_choice_loss
from src.data import build_dataset_and_collator, make_dataloader
from src.lora_utils import load_trainable_state_dict


def evaluate_global_model(model: torch.nn.Module, tokenizer, model_family: str, global_trainable_state: dict[str, torch.Tensor], eval_examples: list[dict[str, any]], cfg: dict[str, any]) -> dict[str, float]:
    device = next(model.parameters()).device
    load_trainable_state_dict(model, global_trainable_state, device)
    model.eval()

    dataset, collator = build_dataset_and_collator(model_family=model_family, examples=eval_examples, tokenizer=tokenizer, max_length=int(cfg['max_length']))
    loader = make_dataloader(dataset=dataset, collator=collator, batch_size=int(cfg['per_device_eval_batch_size']), shuffle=False)

    total_loss = 0.0
    total_count = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            if model_family == 'encoder_mc':
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                logits = outputs.logits
            else:
                loss, logits = _causal_multiple_choice_loss(model, batch)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch['labels']).sum().item())
            total_count += int(batch['labels'].numel())
            total_loss += float(loss.item()) * int(batch['labels'].numel())

    return {'eval_loss': total_loss / max(total_count, 1), 'eval_accuracy': correct / max(total_count, 1)}
