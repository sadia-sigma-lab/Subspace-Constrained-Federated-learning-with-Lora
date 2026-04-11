from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoModelForMultipleChoice, AutoTokenizer, BitsAndBytesConfig

from src.utils import count_trainable_parameters


def infer_model_family(model_name: str) -> str:
    name = model_name.lower()
    if "roberta" in name or "deberta" in name or "bert" in name:
        return "encoder_mc"
    return "causal_mc"


def default_target_modules(model_name: str) -> list[str]:
    name = model_name.lower()
    if "roberta" in name or "deberta" in name or "bert" in name:
        return ["query", "key", "value", "dense"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def make_lora_config(cfg: dict[str, Any], model_family: str) -> LoraConfig:
    task_type = TaskType.SEQ_CLS if model_family == "encoder_mc" else TaskType.CAUSAL_LM
    return LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        bias="none",
        task_type=task_type,
        target_modules=cfg.get("target_modules") or default_target_modules(cfg["model_name"]),
    )


def create_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def create_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_name = cfg["model_name"]
    model_family = infer_model_family(model_name)

    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=getattr(torch, str(cfg.get("bnb_compute_dtype", "float16"))),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_double_quant", True)),
        )

    if model_family == "encoder_mc":
        model = AutoModelForMultipleChoice.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, str(cfg.get("torch_dtype", "float32"))),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=None if load_in_4bit else getattr(torch, str(cfg.get("torch_dtype", "float16"))),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if bool(cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = get_peft_model(model, make_lora_config(cfg, model_family))

    if model_family == "encoder_mc":
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / max(total,1):.4f}%)")
    return model


def create_model_and_tokenizer(cfg: dict[str, Any]):
    tokenizer = create_tokenizer(cfg["model_name"])
    model = create_model(cfg)
    model_family = infer_model_family(cfg["model_name"])
    return model, tokenizer, model_family
