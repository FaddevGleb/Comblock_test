"""Add automatically selected channel examples to style masks for v26."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import regex


ROOT = Path(__file__).resolve().parent
WORD = regex.compile(r"[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)?")


def words(text: str) -> int:
    return len(WORD.findall(text))


def main() -> None:
    config = json.loads((ROOT / "channels_v14.json").read_text(encoding="utf-8"))
    corpus = [json.loads(line) for line in (ROOT / "data" / "telegram_news_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    output = ROOT / "prepared_v26"
    output.mkdir(exist_ok=True)
    audit = {}
    for entry in config["channels"]:
        channel = entry["name"]
        mask = json.loads((ROOT / "prepared_v19" / f"mask_{channel}.json").read_text(encoding="utf-8"))
        posts = [item.get("telegram", {}).get("text", "").strip() for item in corpus if item.get("channel") == channel]
        posts = [post for post in posts if words(post) >= 8 and words(post) <= 90]
        lengths = [words(post) for post in posts]
        quantiles = [statistics.quantiles(lengths, n=10, method="inclusive")[q] for q in (0, 4, 8)] if len(lengths) >= 10 else [min(lengths), statistics.median(lengths), max(lengths)]
        selected = []
        for target in quantiles:
            selected.append(min(posts, key=lambda post: abs(words(post) - target)))
        unique = []
        for post in selected:
            if post not in unique:
                unique.append(post)
        mask["few_shot_style_examples_v26"] = [
            {"text": post, "word_count": words(post), "purpose": purpose}
            for post, purpose in zip(unique, ("short_composition", "typical_composition", "long_composition"))
        ]
        (output / f"train_{channel}.jsonl").write_text((ROOT / "prepared_v19" / f"train_{channel}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (output / f"mask_{channel}.json").write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
        audit[channel] = {"style_corpus": len(posts), "few_shot_examples": len(unique), "word_counts": [words(post) for post in unique]}
    (output / "audit_v26.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
