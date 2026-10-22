"""v14 rerun worker with non-empty adapter checkpoint verification."""

from __future__ import annotations

import json
import gc
from pathlib import Path

import torch
from transformers import Trainer

import dual_qlora_worker_v8 as v8


worker = v8.worker


def train_final(rows: list[dict[str, str]], channel: str, device: int, root: Path, epochs: int, style_mask: dict) -> None:
    model, tokenizer = worker.load_model_and_tokenizer(device)
    model = worker.get_peft_model(
        model,
        worker.LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, use_rslora=True,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    output = root.parent.parent / "adapters" / channel
    trainer = Trainer(
        model=model,
        args=worker.compact_training_args(root / "final_checkpoint", epochs),
        train_dataset=worker.CausalDataset(rows, tokenizer, channel, style_mask),
        data_collator=worker.Collator(tokenizer),
    )
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)

    # Save through PEFT explicitly. This avoids relying only on Trainer's
    # wrapper when the base model is loaded in 4-bit mode.
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "style_mask.json").write_text(json.dumps(style_mask, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_config.json").write_text(
        json.dumps({"model": worker.MODEL_ID, "channel": channel, "epochs": epochs, "max_length": worker.MAX_LENGTH, "seed": worker.SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checkpoint = output / "adapter_model.safetensors"
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"{channel}: adapter checkpoint was not saved or is empty: {checkpoint}")
    if not (output / "adapter_config.json").exists():
        raise RuntimeError(f"{channel}: adapter_config.json was not saved")

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


worker.train_final = train_final


if __name__ == "__main__":
    worker.main()
