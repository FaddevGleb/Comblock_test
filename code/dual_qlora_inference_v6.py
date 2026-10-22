"""v15 inference with identical training prompts and explicit style scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dual_qlora_inference_v4 as implementation
import dual_qlora_inference_v5 as previous
import news_style_runtime_v15 as runtime


implementation.runtime = runtime


def enrich(path: Path, masks: dict[str, dict]) -> tuple[int, float]:
    rows = implementation.read_jsonl(path)
    if not rows:
        raise RuntimeError(f"Empty inference result: {path}")
    scores = []
    for row in rows:
        score = runtime.blind_style_score(row["source_text"], row["response"], masks[row["channel"]])
        row["blind_style"] = score
        scores.append(score["score"])
    implementation.write_jsonl(path, rows)
    return len(rows), sum(scores) / len(scores)


def main() -> None:
    work, channels = previous.adapter_paths_from_argv(sys.argv[1:])
    implementation.main()
    previous.validate_outputs(work, channels)
    fold_masks = {channel: json.loads((work / "workers" / channel / "fold_style_mask.json").read_text(encoding="utf-8")) for channel in channels}
    final_masks = {channel: json.loads((work / "adapters" / channel / "style_mask.json").read_text(encoding="utf-8")) for channel in channels}
    validation = implementation.read_jsonl(work / "validation_results.jsonl")
    for row in validation:
        row["blind_style"] = runtime.blind_style_score(row["source_text"], row["response"], fold_masks[row["channel"]])
    implementation.write_jsonl(work / "validation_results.jsonl", validation)
    counts = {}
    counts["test"], test_style = enrich(work / "test_predictions.jsonl", final_masks)
    for channel in channels:
        counts[channel], _ = enrich(work / "full_generation" / f"{channel}_full_generation.jsonl", final_masks)
    summary_path = work / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["blind_style_by_channel"] = {
        channel: round(sum(row["blind_style"]["score"] for row in validation if row["channel"] == channel) / sum(row["channel"] == channel for row in validation), 4)
        for channel in channels
    }
    summary["test_blind_style_mean"] = round(test_style, 4)
    summary["artifact_rows"] = {"validation": len(validation), **counts}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    previous.validate_outputs(work, channels)
    print("v15 complete", json.dumps(summary["artifact_rows"], ensure_ascii=False))


if __name__ == "__main__":
    main()
