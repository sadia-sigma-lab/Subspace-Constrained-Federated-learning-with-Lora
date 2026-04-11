import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def append_jsonl(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    return float(math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)))


def parse_override_value(raw: str):
    low = raw.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'none', 'null'}:
        return None
    try:
        if raw.startswith('0') and raw != '0' and not raw.startswith('0.'):
            raise ValueError
        return int(raw)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        pass
    return raw


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override must look like key=value, got: {item}')
        key, raw = item.split('=', 1)
        value = parse_override_value(raw)
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return cfg
