"""Convert Telegram JSONL to LUAR author-history JSONL (standard library only)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def clean_text(text: str, channels: list[str]) -> str:
    """Remove explicit source identities, preserving punctuation and emoji."""
    text = unicodedata.normalize("NFKC", text).replace("\u200b", "")
    text = re.sub(r"(?:https?://|www\.|(?:t|telegram)\.me/)\S+", " ", text, flags=re.I)
    text = re.sub(r"(?<!\w)@[A-Za-z0-9_]+", " ", text)
    for channel in channels:
        text = re.sub(r"(?<!\w)#?" + re.escape(channel) + r"(?!\w)", " ", text, flags=re.I)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fingerprint(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def shingles(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def deduplicate(posts: list[dict], threshold: float) -> tuple[list[dict], dict]:
    """Global exact + word-5-shingle Jaccard dedup, before any splitting."""
    seen_ids, seen_texts = set(), set()
    index: dict[str, set[int]] = defaultdict(set)
    kept, signatures, removed = [], [], Counter()
    for post in sorted(posts, key=lambda p: (p["published_at"], p["id"])):
        signature = fingerprint(post["text"])
        if post["id"] in seen_ids or signature in seen_texts:
            removed["exact_duplicate"] += 1
            continue
        seen_ids.add(post["id"])
        seen_texts.add(signature)
        sh = shingles(signature)
        candidates = set().union(*(index[s] for s in sh)) if sh else set()
        if threshold < 1 and any(
            len(sh & signatures[i]) / len(sh | signatures[i]) >= threshold
            for i in candidates
        ):
            removed["near_duplicate"] += 1
            continue
        i = len(kept)
        kept.append(post)
        signatures.append(sh)
        for s in sh:
            index[s].add(i)
    return kept, dict(removed)


def split_posts(posts: list[dict], args) -> tuple[dict, dict]:
    channels: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        channels[p["channel"]].append(p)
    result = {s: {} for s in SPLITS}
    dropped = {}
    if args.split_mode == "channel":
        names = sorted(channels)
        random.Random(args.seed).shuffle(names)
        # At least two channel identities per split for meaningful negatives.
        if len(names) < 6:
            raise ValueError("Channel-disjoint split requires >=6 channels. Use --split-mode temporal.")
        nv = max(2, round(len(names) * args.val_fraction))
        nt = max(2, round(len(names) * args.test_fraction))
        if len(names) - nv - nt < 2:
            raise ValueError("Not enough train channels after splitting; reduce split fractions.")
        allocations = {"validation": names[:nv], "test": names[nv:nv + nt], "train": names[nv + nt:]}
        for split, names_in_split in allocations.items():
            for name in names_in_split:
                required = 2 * args.episode_length
                if len(channels[name]) < required:
                    dropped[name] = {"reason": f"fewer than {required} posts"}
                else:
                    result[split][name] = channels[name]
    else:
        for name, history in channels.items():
            history = sorted(history, key=lambda p: (p["published_at"], p["id"]))
            n = len(history)
            a = int(n * (1 - args.val_fraction - args.test_fraction))
            b = int(n * (1 - args.test_fraction))
            parts = {"train": history[:a], "validation": history[a:b], "test": history[b:]}
            # Drop tail of earlier partitions; equal timestamps cannot cross boundaries.
            if parts["validation"] and parts["test"]:
                for earlier, later in (("train", "validation"), ("validation", "test")):
                    cutoff = parse_time(parts[later][0]["published_at"]) - timedelta(hours=args.embargo_hours)
                    parts[earlier] = [p for p in parts[earlier] if parse_time(p["published_at"]) < cutoff]
            counts = {s: len(v) for s, v in parts.items()}
            if min(counts.values()) < 2 * args.episode_length:
                dropped[name] = {"reason": "too few posts after time split / embargo", "counts": counts}
                continue
            for s, items in parts.items():
                result[s][name] = items
    for split in SPLITS:
        if len(result[split]) < 2:
            raise ValueError(f"{split}: need >=2 channels; filtered channels: {dropped}")
    return result, dropped


def history_row(channel: str, posts: list[dict], channel_ids: dict) -> dict:
    return {
        "author_id": channel_ids[channel], "channel": channel,
        "syms": [p["text"] for p in posts],
        "post_ids": [p["id"] for p in posts],
        "published_at": [p["published_at"] for p in posts],
    }


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args) -> dict:
    if not (0 < args.val_fraction < 0.5 and 0 < args.test_fraction < 0.5
            and args.val_fraction + args.test_fraction < 1):
        raise ValueError("val/test fractions must be >0, <0.5, and sum to <1")
    if args.episode_length < 1 or args.embargo_hours < 0 or not 0 < args.near_duplicate_threshold <= 1:
        raise ValueError("Invalid episode length, embargo or duplicate threshold")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.output_dir}")
    rows = []
    with args.input.open(encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {number}") from exc
    channels = sorted({r["channel"] for r in rows if isinstance(r.get("channel"), str) and r["channel"]})
    posts, rejected = [], Counter()
    for number, row in enumerate(rows):
        telegram = row.get("telegram") or {}
        if not row.get("channel") or not isinstance(telegram.get("text"), str):
            rejected["missing_channel_or_text"] += 1
            continue
        text = clean_text(telegram["text"], channels)
        if len(re.findall(r"\w+", text)) < args.min_words:
            rejected["short_text"] += 1
            continue
        try:
            date = parse_time(telegram["published_at"]).isoformat()
        except (ValueError, TypeError, KeyError):
            rejected["missing_or_invalid_time"] += 1
            continue
        post_id = row.get("id") or f"{row['channel']}:{telegram.get('message_id', number)}"
        posts.append({"id": str(post_id), "channel": row["channel"], "text": text, "published_at": date})
    posts, duplicate_stats = deduplicate(posts, args.near_duplicate_threshold)
    split, dropped = split_posts(posts, args)
    kept_names = sorted(set().union(*(set(v) for v in split.values())))
    channel_ids = {name: i for i, name in enumerate(kept_names)}
    files, audit = {}, []
    for s in SPLITS:
        histories = split[s]
        if s == "train":
            files["train.jsonl"] = [history_row(c, histories[c], channel_ids) for c in sorted(histories)]
            files["data.jsonl"] = files["train.jsonl"]  # Native raw_all Reddit loader filename.
        else:
            query_rows, target_rows = [], []
            for c in sorted(histories):
                items = histories[c]
                midpoint = len(items) // 2
                # Older half is the gallery; newer half is the query set.
                target_rows.append(history_row(c, items[:midpoint], channel_ids))
                query_rows.append(history_row(c, items[midpoint:], channel_ids))
            files[f"{s}_queries.jsonl"] = query_rows
            files[f"{s}_targets.jsonl"] = target_rows
        for c, items in histories.items():
            audit.extend({"id": p["id"], "channel": c, "split": s, "published_at": p["published_at"],
                          "clean_sha256": hashlib.sha256(p["text"].encode()).hexdigest()} for p in items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in files.items():
        write_jsonl(args.output_dir / name, records)
    write_jsonl(args.output_dir / "post_split_manifest.jsonl", audit)
    report = {
        "schema_version": 1, "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_posts": len(rows), "rejected": dict(rejected), "duplicates": duplicate_stats,
        "dropped_channels": dropped, "channel_to_author_id": channel_ids,
        "split_mode": args.split_mode, "embargo_hours": args.embargo_hours,
        "episode_length": args.episode_length, "seed": args.seed,
        "counts": {s: {c: len(v) for c, v in split[s].items()} for s in SPLITS},
        "files": {n: {"histories": len(rs), "posts": sum(len(r["syms"]) for r in rs)} for n, rs in files.items()},
        "limitations": [
            "Channel is a source label, not necessarily an individual author.",
            "Temporal split measures known-channel performance; it does not measure unseen-channel generalization.",
            "Topics, named entities, boilerplate and approximate paraphrases can still reveal source identity.",
            "Do not interpret ROC-AUC as proof that only style was learned.",
        ],
    }
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "telegram_news_dataset.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "luar_telegram_data")
    p.add_argument("--split-mode", choices=("temporal", "channel"), default="temporal")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--embargo-hours", type=float, default=24)
    p.add_argument("--episode-length", type=int, default=4)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    cli = parser()
    try:
        result = prepare(cli.parse_args())
    except ValueError as exc:
        cli.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
