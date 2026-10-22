"""v18 two-GPU worker: r=32, measured masks, source-grounded validation."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer

import news_style_runtime_v18 as runtime
import dual_qlora_worker_v11 as previous

worker = previous.worker
worker.runtime = runtime
previous.runtime = runtime
worker.MAX_EPOCHS = 4
worker.MAX_LENGTH = 1536


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    required = {"instruction", "input", "output"}
    if len(rows) < 20 or any(not required.issubset(row) or not all(str(row[key]).strip() for key in required) for row in rows):
        raise ValueError(f"Invalid or insufficient training rows: {path}")
    if len({(row["input"], row["output"]) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate training pairs: {path}")
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    supervised = [i for i, row in enumerate(rows) if row.get("origin") == "supervised"]
    matched = [i for i, row in enumerate(rows) if row.get("origin") == "matched_news"]
    selected = []
    for pool in (supervised, matched):
        ordered = sorted(pool, key=lambda i: len(runtime.words(rows[i]["output"])))
        if len(ordered) >= 2:
            selected.extend([ordered[len(ordered) // 3], ordered[(2 * len(ordered)) // 3]])
    if len(selected) != 4:
        raise RuntimeError("Need two supervised and two matched validation pairs")
    indices = set(selected)
    return [row for i, row in enumerate(rows) if i not in indices], [row for i, row in enumerate(rows) if i in indices]


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, use_rslora=True,
        bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def train_split(train_rows: list[dict], val_rows: list[dict], channel: str, device: int, root: Path, scorer, mask: dict) -> None:
    mask = runtime.fold_mask(mask, train_rows)
    (root / "fold_style_mask.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    model, tokenizer = worker.load_model_and_tokenizer(device)
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=worker.compact_training_args(root / "checkpoints" / "split", worker.MAX_EPOCHS),
        train_dataset=worker.CausalDataset(train_rows, tokenizer, channel, mask),
        data_collator=worker.Collator(tokenizer),
        callbacks=[worker.GenerationCallback(val_rows, channel, mask, tokenizer, device, scorer, root / "metrics")],
    )
    trainer.train()
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def train_final(rows: list[dict], channel: str, device: int, root: Path, epochs: int, mask: dict) -> None:
    model, tokenizer = worker.load_model_and_tokenizer(device)
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    destination = root.parent.parent / "adapters" / channel
    trainer = Trainer(
        model=model,
        args=worker.compact_training_args(root / "final_checkpoint", epochs),
        train_dataset=worker.CausalDataset(rows, tokenizer, channel, mask),
        data_collator=worker.Collator(tokenizer),
    )
    trainer.train()
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "style_mask.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "training_config.json").write_text(
        json.dumps({"channel": channel, "rank": 32, "lora_alpha": 64, "epochs": epochs, "model": worker.MODEL_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tensor = destination / "adapter_model.safetensors"
    if not tensor.exists() or tensor.stat().st_size < 1024:
        raise RuntimeError(f"Adapter missing: {destination}")
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


worker.split_rows = split_rows
worker.read_jsonl = read_jsonl
worker.train_split = train_split
worker.train_final = train_final

if __name__ == "__main__":
    worker.main()
