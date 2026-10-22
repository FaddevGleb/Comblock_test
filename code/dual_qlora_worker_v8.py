"""Generic v14 channel worker. The adapter name and files arrive as arguments."""

from __future__ import annotations

import json
from pathlib import Path

import news_style_runtime_v14 as runtime
import dual_qlora_worker_v6 as worker


worker.runtime = runtime
worker.MAX_EPOCHS = 6
worker.MAX_LENGTH = 1280


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) < 12:
        raise ValueError(f"At least 12 audited pairs are needed: {path}")
    if any(not all(row.get(key) for key in ("instruction", "input", "output")) for row in rows):
        raise ValueError(f"Missing instruction/input/output: {path}")
    return rows


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    # Hold out supervised examples first, then matched-news examples if present.
    supervised = [i for i, row in enumerate(rows) if row.get("origin") == "supervised"]
    matched = [i for i, row in enumerate(rows) if row.get("origin") != "supervised"]

    def evenly_spaced(indices: list[int], count: int) -> list[int]:
        if not indices or count <= 0:
            return []
        indices = sorted(indices, key=lambda i: len(runtime.words(rows[i]["output"])))
        count = min(count, len(indices))
        return [indices[int((j + 0.5) * len(indices) / count)] for j in range(count)]

    selected = set(evenly_spaced(supervised, 5) + evenly_spaced(matched, 2))
    if len(selected) < 3:
        selected = set(evenly_spaced(list(range(len(rows))), 3))
    return [row for i, row in enumerate(rows) if i not in selected], [row for i, row in enumerate(rows) if i in selected]


worker.read_jsonl = read_jsonl
worker.split_rows = split_rows


def aggregate_and_select(root: Path, channel: str, val_rows: list[dict], scorer) -> int:
    metrics_dir = root / "metrics"
    mask = runtime.load_mask(root.parent.parent / "assets" / f"mask_{channel}.json")
    epochs = []
    for epoch in range(1, worker.MAX_EPOCHS + 1):
        path = metrics_dir / f"split_epoch_{epoch}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(val_rows):
            continue
        metrics = worker.metric_dict([r["prediction"] for r in rows], [r["reference"] for r in rows], scorer)
        checks = [runtime.factual_check(r["input"], r["prediction"]) for r in rows]
        metrics["hard_fact_error_rate"] = round(sum(not check["deterministic_passed"] for check in checks) / len(rows), 6)
        style_scores = []
        for row in rows:
            features = runtime.style_comparison(row["prediction"], mask)["features"]
            length = features["word_count"]
            range_ = mask["word_count"]
            paragraph_target = mask["paragraph_count"]["median"]
            emoji_rate = mask["emoji"]["starts_with_emoji_rate"]
            score = [range_["p10"] <= length <= range_["p90"], abs(features["paragraph_count"] - paragraph_target) <= 1]
            if emoji_rate >= 0.8:
                score.append(features["starts_with_emoji"])
            elif emoji_rate <= 0.2:
                score.append(not features["starts_with_emoji"])
            signature = mask.get("signature", {})
            if signature.get("rate", 0) >= 0.8:
                score.append(signature["value"] in row["prediction"])
            style_scores.append(sum(score) / len(score))
        metrics["style_adherence"] = round(sum(style_scores) / len(style_scores), 6)
        metrics["selection_score"] = round(
            0.25 * metrics["chrf"] / 100 + 0.20 * metrics["rougeL_f1"]
            + 0.20 * metrics["style_adherence"] + 0.35 * (1 - metrics["hard_fact_error_rate"]), 6
        )
        metrics["epoch"] = epoch
        epochs.append(metrics)
    if not epochs:
        raise RuntimeError(f"{channel}: no complete epoch metrics")
    best = max(epochs, key=lambda row: (row["selection_score"], -row["epoch"]))
    (metrics_dir / "validation_by_epoch.json").write_text(json.dumps(epochs, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "summary.json").write_text(json.dumps({"channel": channel, "best_epoch": best["epoch"], "metrics": best}, ensure_ascii=False, indent=2), encoding="utf-8")
    return best["epoch"]


worker.aggregate_and_select = aggregate_and_select


if __name__ == "__main__":
    worker.main()
