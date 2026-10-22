"""v19 worker: all-linear LoRA rank 16, lower LR, original pairs only."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

import dual_qlora_worker_v12 as previous
import news_style_runtime_v18 as runtime


worker = previous.worker
worker.runtime = runtime
previous.runtime = runtime
worker.MAX_EPOCHS = 5
worker.MAX_LENGTH = 1536


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, use_rslora=True,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )


def training_args(output_dir: Path, epochs: int) -> TrainingArguments:
    args = previous.worker.compact_training_args(output_dir, epochs)
    args.learning_rate = 1e-5
    return args


def train_split(train_rows, val_rows, channel, device, root, scorer, mask):
    mask = runtime.fold_mask(mask, train_rows)
    (root / "fold_style_mask.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    model, tokenizer = worker.load_model_and_tokenizer(device)
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model, args=training_args(root / "checkpoints" / "split", worker.MAX_EPOCHS),
        train_dataset=worker.CausalDataset(train_rows, tokenizer, channel, mask),
        data_collator=worker.Collator(tokenizer),
        callbacks=[worker.GenerationCallback(val_rows, channel, mask, tokenizer, device, scorer, root / "metrics")],
    )
    trainer.train()
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def train_final(rows, channel, device, root, epochs, mask):
    mask = runtime.fold_mask(mask, rows)
    model, tokenizer = worker.load_model_and_tokenizer(device)
    model = get_peft_model(model, lora_config())
    destination = root.parent.parent / "adapters" / channel
    trainer = Trainer(
        model=model, args=training_args(root / "final_checkpoint", epochs),
        train_dataset=worker.CausalDataset(rows, tokenizer, channel, mask),
        data_collator=worker.Collator(tokenizer),
    )
    trainer.train()
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    (destination / "style_mask.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "training_config.json").write_text(json.dumps({
        "channel": channel, "rank": 16, "lora_alpha": 32, "target_modules": "all-linear",
        "learning_rate": 1e-5, "epochs": epochs, "model": worker.MODEL_ID,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    tensor = destination / "adapter_model.safetensors"
    if not tensor.exists() or tensor.stat().st_size < 1024:
        raise RuntimeError(f"Adapter missing: {destination}")
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


worker.lora_config = lora_config
def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(range(len(rows)), key=lambda i: len(runtime.words(rows[i]["output"])))
    selected = {ordered[len(ordered) // 4], ordered[len(ordered) // 2], ordered[(3 * len(ordered)) // 4]}
    return [row for i, row in enumerate(rows) if i not in selected], [row for i, row in enumerate(rows) if i in selected]


worker.split_rows = split_rows
worker.train_split = train_split
worker.train_final = train_final


if __name__ == "__main__":
    worker.main()
