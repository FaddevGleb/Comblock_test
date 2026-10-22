"""v15 worker: fold-only style examples and separate fact/style epoch selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import news_style_runtime_v15 as runtime
import dual_qlora_worker_v9 as previous


worker = previous.worker
worker.runtime = runtime
worker.MAX_EPOCHS = 5
worker.MAX_LENGTH = 1280


def train_split(train_rows: list[dict], val_rows: list[dict], channel: str, device: int, root: Path, scorer, style_mask: dict) -> None:
    mask = runtime.fold_mask(style_mask, train_rows)
    (root / "fold_style_mask.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    worker._v15_train_split_original(train_rows, val_rows, channel, device, root, scorer, mask)


worker._v15_train_split_original = worker.train_split
worker.train_split = train_split


def train_final(rows: list[dict], channel: str, device: int, root: Path, epochs: int, style_mask: dict) -> None:
    final_mask = runtime.fold_mask(style_mask, rows)
    previous.train_final(rows, channel, device, root, epochs, final_mask)


worker.train_final = train_final


def aggregate_and_select(root: Path, channel: str, val_rows: list[dict], scorer) -> int:
    metrics_dir = root / "metrics"
    mask = json.loads((root / "fold_style_mask.json").read_text(encoding="utf-8"))
    epochs = []
    for epoch in range(1, worker.MAX_EPOCHS + 1):
        path = metrics_dir / f"split_epoch_{epoch}.jsonl"
        if not path.exists():
            raise RuntimeError(f"{channel}: missing validation predictions for epoch {epoch}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(val_rows):
            raise RuntimeError(f"{channel}: incomplete validation epoch {epoch}")
        metrics = worker.metric_dict([r["prediction"] for r in rows], [r["reference"] for r in rows], scorer)
        checks = [runtime.factual_check(row["input"], row["prediction"]) for row in rows]
        style = [runtime.blind_style_score(row["input"], row["prediction"], mask) for row in rows]
        metrics.update({
            "epoch": epoch,
            "hard_fact_error_rate": round(sum(not c["deterministic_passed"] for c in checks) / len(rows), 6),
            "blind_style_score": round(sum(value["score"] for value in style) / len(style), 6),
        })
        metrics["semantic_quality"] = round(0.55 * metrics["chrf"] / 100 + 0.45 * metrics["rougeL_f1"], 6)
        epochs.append(metrics)

    # First constrain hard fact errors and semantic regression, then select
    # the best reference-blind author-style score among eligible epochs.
    safest = min(row["hard_fact_error_rate"] for row in epochs)
    semantic_best = max(row["semantic_quality"] for row in epochs)
    eligible = [row for row in epochs if row["hard_fact_error_rate"] <= safest + 0.05 and row["semantic_quality"] >= semantic_best - 0.08]
    if not eligible:
        raise RuntimeError(f"{channel}: no eligible epochs")
    best = max(eligible, key=lambda row: (row["blind_style_score"], row["semantic_quality"], -row["epoch"]))
    (metrics_dir / "validation_by_epoch.json").write_text(json.dumps(epochs, ensure_ascii=False, indent=2), encoding="utf-8")
    with (metrics_dir / "validation_by_epoch.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
        writer.writeheader()
        writer.writerows(epochs)
    (metrics_dir / "summary.json").write_text(json.dumps({
        "channel": channel, "best_epoch": best["epoch"], "metrics": best,
        "selection": "reference-blind style score, subject to factual and semantic gates",
        "semantic_best": semantic_best, "minimum_hard_fact_error_rate": safest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(best["epoch"])


worker.aggregate_and_select = aggregate_and_select


if __name__ == "__main__":
    worker.main()
