"""Single-channel worker used by KAGGLE_DUAL_T4_QLORA.ipynb.

The notebook launches one worker per GPU.  The worker performs 5-fold
generation-based validation and then trains one final adapter on all examples.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import regex
import torch
from bert_score import BERTScorer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import style_runtime_v3 as runtime


MODEL_ID = runtime.MODEL_ID
MAX_LENGTH = 1024
MAX_EPOCHS = 8
SEED = 42
TRAIN_SIZE = 23
VAL_SIZE = 2
WORD_RE = regex.compile(r"[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)?")
GRAPHEME_RE = regex.compile(r"\X")
EMOJI_RE = regex.compile(r"[\p{Extended_Pictographic}\p{Regional_Indicator}]")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d[\d\s]*(?:[.,]\d+)?(?:\s*%)?")


class RussianTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return WORD_RE.findall(text.casefold())


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    required = {"instruction", "input", "output"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"Invalid schema in {path}")
    if len(rows) != 25:
        raise ValueError(f"Expected 25 rows in {path}, got {len(rows)}")
    if any(not all(str(row[key]).strip() for key in required) for row in rows):
        raise ValueError(f"Empty field in {path}")
    return rows


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministic validation with one short and one long post per channel."""
    if len(rows) != TRAIN_SIZE + VAL_SIZE:
        raise ValueError(f"Expected {TRAIN_SIZE + VAL_SIZE} rows")
    order = sorted(range(len(rows)), key=lambda i: len(words(rows[i]["output"])))
    val_positions = {6, 18}
    val_indices = {order[position] for position in val_positions}
    train_rows = [row for i, row in enumerate(rows) if i not in val_indices]
    val_rows = [row for i, row in enumerate(rows) if i in val_indices]
    return train_rows, val_rows


def words(text: str) -> list[str]:
    return [item.casefold() for item in WORD_RE.findall(text)]


def emoji_list(text: str) -> list[str]:
    return [cluster for cluster in GRAPHEME_RE.findall(text) if EMOJI_RE.search(cluster)]


def messages(row: dict[str, str], channel: str, style_mask: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return runtime.messages(row, channel, style_mask)


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, channel: str, style_mask: dict[str, Any]):
        self.items = []
        for row in rows:
            prompt, full = messages(row, channel, style_mask)
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("Chat-template prefix mismatch; assistant loss mask is unsafe")
            if len(full_ids) >= MAX_LENGTH:
                raise ValueError(f"Example exceeds MAX_LENGTH={MAX_LENGTH}")
            eos_id = tokenizer.eos_token_id
            if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
                full_ids.append(eos_id)
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.items.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class Collator:
    def __init__(self, tokenizer: Any):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(item["input_ids"]) for item in batch)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            padding = length - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_id] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def f1_from_counters(predicted: Counter[str], reference: Counter[str]) -> dict[str, float]:
    overlap = sum((predicted & reference).values())
    precision = overlap / max(1, sum(predicted.values()))
    recall = overlap / max(1, sum(reference.values()))
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}


def metric_dict(predictions: list[str], references: list[str], scorer: Any | None) -> dict[str, float]:
    bleu = BLEU(tokenize="intl").corpus_score(predictions, [references]).score
    chrf = CHRF(word_order=2).corpus_score(predictions, [references]).score
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=RussianTokenizer(), use_stemmer=False)
    rouge_scores = [rouge.score(reference, prediction) for prediction, reference in zip(predictions, references)]
    numbers = f1_from_counters(Counter(x for text in predictions for x in NUMBER_RE.findall(text)), Counter(x for text in references for x in NUMBER_RE.findall(text)))
    emojis = f1_from_counters(Counter(x for text in predictions for x in emoji_list(text)), Counter(x for text in references for x in emoji_list(text)))
    result = {
        "bleu": float(bleu),
        "chrf": float(chrf),
        "rouge1_f1": float(np.mean([x["rouge1"].fmeasure for x in rouge_scores])),
        "rouge2_f1": float(np.mean([x["rouge2"].fmeasure for x in rouge_scores])),
        "rougeL_f1": float(np.mean([x["rougeL"].fmeasure for x in rouge_scores])),
        "number_precision": numbers["precision"], "number_recall": numbers["recall"], "number_f1": numbers["f1"],
        "emoji_precision": emojis["precision"], "emoji_recall": emojis["recall"], "emoji_f1": emojis["f1"],
        "length_ratio": float(np.mean([len(words(p)) / max(1, len(words(r))) for p, r in zip(predictions, references)])),
        "empty_rate": sum(not p.strip() for p in predictions) / max(1, len(predictions)),
        "truncated_rate": sum(len(words(p)) >= 250 for p in predictions) / max(1, len(predictions)),
    }
    if scorer is not None:
        precision, recall, f1 = scorer.score(predictions, references)
        result.update({"bertscore_precision": float(precision.mean()), "bertscore_recall": float(recall.mean()), "bertscore_f1": float(f1.mean())})
    else:
        result.update({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
    result["quality_score"] = (0.20 * result["bleu"] / 100 + 0.20 * result["chrf"] / 100 + 0.20 * result["rougeL_f1"] + 0.25 * result["bertscore_f1"] + 0.15 * result["number_f1"])
    return {key: round(float(value), 6) for key, value in result.items()}


def compact_training_args(output_dir: Path, epochs: int) -> TrainingArguments:
    common = dict(output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=5e-5, per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.10, weight_decay=0.01, max_grad_norm=1.0, logging_steps=1, save_strategy="no", report_to="none", fp16=True, remove_unused_columns=False, seed=SEED)
    names = TrainingArguments.__init__.__code__.co_varnames
    common["eval_strategy" if "eval_strategy" in names else "evaluation_strategy"] = "no"
    return TrainingArguments(**common)


def load_model_and_tokenizer(device: int) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    current_device = torch.cuda.current_device()
    config = AutoConfig.from_pretrained(MODEL_ID)
    if hasattr(config, "use_sliding_window"):
        config.use_sliding_window = False
    if hasattr(config, "sliding_window"):
        config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, quantization_config=quantization, torch_dtype=torch.float16, device_map={"": current_device}, attn_implementation="eager")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def generate_predictions(model: Any, tokenizer: Any, rows: list[dict[str, str]], channel: str, style_mask: dict[str, Any], device: int) -> list[str]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for row in rows:
            prompt, _ = messages(row, channel, style_mask)
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(f"cuda:{device}")
            generated = model.generate(**batch, max_new_tokens=256, do_sample=False, num_beams=1, repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            continuation = generated[0, batch["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            guarded_output, _ = runtime.guard_output(row["input"], raw_output, channel, style_mask)
            outputs.append(guarded_output)
    model.config.use_cache = False
    model.train()
    return outputs


class GenerationCallback(TrainerCallback):
    def __init__(self, rows: list[dict[str, str]], channel: str, style_mask: dict[str, Any], tokenizer: Any, device: int, scorer: Any, out_dir: Path):
        self.rows, self.channel, self.style_mask, self.tokenizer = rows, channel, style_mask, tokenizer
        self.device, self.scorer, self.out_dir = device, scorer, out_dir

    def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        predictions = generate_predictions(model, self.tokenizer, self.rows, self.channel, self.style_mask, self.device)
        references = [row["output"] for row in self.rows]
        metrics = metric_dict(predictions, references, self.scorer)
        epoch = int(round(state.epoch or 0))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / f"split_epoch_{epoch}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.out_dir / f"split_epoch_{epoch}.jsonl").open("w", encoding="utf-8") as stream:
            for row, prediction in zip(self.rows, predictions):
                stream.write(json.dumps({"input": row["input"], "reference": row["output"], "prediction": prediction, "factual_check": runtime.factual_check(row["input"], prediction), "length_plan": runtime.length_plan(row["input"], self.style_mask)}, ensure_ascii=False) + "\n")


def train_split(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], channel: str, device: int, root: Path, scorer: Any, style_mask: dict) -> None:
    model, tokenizer = load_model_and_tokenizer(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, use_rslora=True, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainer = Trainer(model=model, args=compact_training_args(root / "checkpoints" / "split", MAX_EPOCHS), train_dataset=CausalDataset(train_rows, tokenizer, channel, style_mask), data_collator=Collator(tokenizer), callbacks=[GenerationCallback(val_rows, channel, style_mask, tokenizer, device, scorer, root / "metrics")])
    trainer.train()
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def aggregate_and_select(root: Path, channel: str, val_rows: list[dict[str, str]], scorer: Any) -> int:
    metrics_dir = root / "metrics"
    rows_out = []
    for epoch in range(1, MAX_EPOCHS + 1):
        predictions, references = [], []
        path = metrics_dir / f"split_epoch_{epoch}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                predictions.append(item["prediction"])
                references.append(item["reference"])
        if len(predictions) == len(val_rows):
            metrics = metric_dict(predictions, references, scorer)
            metrics["epoch"] = epoch
            rows_out.append(metrics)
    if not rows_out:
        raise RuntimeError(f"No complete OOF metrics for {channel}")
    best = max(rows_out, key=lambda item: (item["quality_score"], -item["epoch"]))
    (root / "metrics" / "validation_by_epoch.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "metrics" / "summary.json").write_text(json.dumps({"channel": channel, "best_epoch": best["epoch"], "metrics": best}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "metrics" / "validation_by_epoch.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    return int(best["epoch"])


def train_final(rows: list[dict[str, str]], channel: str, device: int, root: Path, epochs: int, style_mask: dict) -> None:
    model, tokenizer = load_model_and_tokenizer(device)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, use_rslora=True, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    output = root.parent.parent / "adapters" / channel
    trainer = Trainer(model=model, args=compact_training_args(root / "final_checkpoint", epochs), train_dataset=CausalDataset(rows, tokenizer, channel, style_mask), data_collator=Collator(tokenizer))
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    (output / "style_mask.json").write_text(json.dumps(style_mask, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_config.json").write_text(json.dumps({"model": MODEL_ID, "channel": channel, "epochs": epochs, "max_length": MAX_LENGTH, "seed": SEED}, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--style-mask", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    random.seed(SEED); np.random.seed(SEED); set_seed(SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.data)
    style_mask = runtime.load_mask(args.style_mask)
    train_rows, val_rows = split_rows(rows)
    (args.output / "split.json").write_text(json.dumps({"train_count": len(train_rows), "validation_count": len(val_rows), "validation_indices_by_length": [rows.index(row) for row in val_rows], "validation_examples": [{"input": row["input"], "reference": row["output"]} for row in val_rows]}, ensure_ascii=False, indent=2), encoding="utf-8")
    scorer = BERTScorer(model_type="xlm-roberta-base", lang="ru", device="cpu", rescale_with_baseline=False)
    train_split(train_rows, val_rows, args.channel, args.device, args.output, scorer, style_mask)
    best_epoch = aggregate_and_select(args.output, args.channel, val_rows, scorer)
    train_final(rows, args.channel, args.device, args.output, best_epoch, style_mask)
    (args.output / "worker_done.json").write_text(json.dumps({"channel": args.channel, "best_epoch": best_epoch}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
