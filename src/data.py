from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, Sequence

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def _normalize_text(text: str) -> str:
    return ' '.join(str(text).replace(' [title]', '. ').split())


def format_hellaswag_context(example: Dict[str, Any]) -> str:
    ctx_a = _normalize_text(example['ctx_a'])
    ctx_b = _normalize_text(example['ctx_b'])
    return f"{ctx_a} {ctx_b}".strip()


def get_partition_key(example: Dict[str, Any], field: str) -> str:
    if field in example:
        value = example[field]
        if isinstance(value, list):
            return '|'.join(map(str, value))
        return str(value)
    return str(example['label'])


def load_raw_hellaswag(
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ds = load_dataset('hellaswag')
    train = list(ds['train'])
    valid = list(ds['validation'])

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(valid)

    if max_train_samples is not None:
        train = train[:max_train_samples]
    if max_eval_samples is not None:
        valid = valid[:max_eval_samples]
    return train, valid


def iid_partition(items: Sequence[dict[str, Any]], num_clients: int, seed: int) -> dict[int, list[dict[str, Any]]]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    parts: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_clients)}
    for idx, item in enumerate(items):
        parts[idx % num_clients].append(item)
    return parts


def dirichlet_partition(
    items: Sequence[dict[str, Any]],
    num_clients: int,
    alpha: float,
    seed: int,
    field: str = 'activity_label',
    min_size: int = 1,
) -> dict[int, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[get_partition_key(item, field)].append(item)

    client_items: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_clients)}

    for examples in buckets.values():
        rng.shuffle(examples)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = np.floor(proportions * len(examples)).astype(int)
        while counts.sum() < len(examples):
            counts[rng.integers(0, num_clients)] += 1

        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            if end > start:
                client_items[client_id].extend(examples[start:end])
            start = end

    small_clients = [cid for cid, exs in client_items.items() if len(exs) < min_size]
    if small_clients:
        all_items = []
        for cid in range(num_clients):
            all_items.extend(client_items[cid])
        rng.shuffle(all_items)
        client_items = {i: [] for i in range(num_clients)}
        for idx, item in enumerate(all_items):
            client_items[idx % num_clients].append(item)

    return client_items


def partition_statistics(client_partitions: dict[int, list[dict[str, Any]]], field: str = 'activity_label') -> dict[str, Any]:
    sizes = {str(cid): len(exs) for cid, exs in client_partitions.items()}
    hist = {
        str(cid): dict(Counter(get_partition_key(ex, field) for ex in exs))
        for cid, exs in client_partitions.items()
    }
    return {
        'num_clients': len(client_partitions),
        'client_sizes': sizes,
        'partition_field': field,
        'label_histograms': hist,
    }


class EncoderHellaSwagDataset(Dataset):
    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        context = format_hellaswag_context(ex)
        return {
            'context': context,
            'choices': list(ex['endings']),
            'label': int(ex['label']),
        }


class EncoderMultipleChoiceCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        first_sentences: list[str] = []
        second_sentences: list[str] = []
        labels: list[int] = []

        for item in batch:
            labels.append(int(item['label']))
            first_sentences.extend([item['context']] * len(item['choices']))
            second_sentences.extend(item['choices'])

        encoded = self.tokenizer(
            first_sentences,
            second_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        num_choices = len(batch[0]['choices'])
        collated = {key: value.view(len(batch), num_choices, -1) for key, value in encoded.items()}
        collated['labels'] = torch.tensor(labels, dtype=torch.long)
        return collated


class CausalMultipleChoiceDataset(Dataset):
    def __init__(self, examples: Sequence[dict[str, Any]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {
            'prompt': format_hellaswag_context(ex),
            'choices': list(ex['endings']),
            'label': int(ex['label']),
        }


class CausalMultipleChoiceCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_choice(self, prompt: str, choice: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        choice_ids = self.tokenizer(' ' + choice, add_special_tokens=False)['input_ids']
        full_ids = (prompt_ids + choice_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        loss_mask = [0] * prompt_len + [1] * max(0, len(full_ids) - prompt_len)
        return full_ids, loss_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids: list[list[int]] = []
        all_attention_mask: list[list[int]] = []
        all_loss_mask: list[list[int]] = []
        labels: list[int] = []

        num_choices = len(batch[0]['choices'])
        for item in batch:
            labels.append(int(item['label']))
            for choice in item['choices']:
                ids, loss_mask = self._build_choice(item['prompt'], choice)
                all_input_ids.append(ids)
                all_attention_mask.append([1] * len(ids))
                all_loss_mask.append(loss_mask)

        padded = self.tokenizer.pad(
            {'input_ids': all_input_ids, 'attention_mask': all_attention_mask},
            padding=True,
            return_tensors='pt',
        )
        max_len = padded['input_ids'].shape[-1]
        padded_loss = [lm + [0] * (max_len - len(lm)) for lm in all_loss_mask]
        loss_mask = torch.tensor(padded_loss, dtype=torch.float32)

        return {
            'input_ids': padded['input_ids'].view(len(batch), num_choices, -1),
            'attention_mask': padded['attention_mask'].view(len(batch), num_choices, -1),
            'loss_mask': loss_mask.view(len(batch), num_choices, -1),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def build_dataset_and_collator(model_family: str, examples: Sequence[dict[str, Any]], tokenizer, max_length: int):
    if model_family == 'encoder_mc':
        return EncoderHellaSwagDataset(examples), EncoderMultipleChoiceCollator(tokenizer, max_length)
    if model_family == 'causal_mc':
        return CausalMultipleChoiceDataset(examples), CausalMultipleChoiceCollator(tokenizer, max_length)
    raise ValueError(f'Unsupported model_family: {model_family}')


def make_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
