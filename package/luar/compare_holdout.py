"""Compare a trained LUAR export and AtesiT on the same channel holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from train_kaggle import compute_metrics, embed, eval_episodes, load_export, read_histories, tokenize_histories


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def atesi_embeddings(rows, tokenizer, model, max_length, episode_length, device, batch_size=32):
    all_embeddings, all_labels = [], []
    for row in rows:
        encoded = tokenizer(row["syms"], padding="max_length", truncation=True,
                            max_length=max_length, return_tensors="pt")
        outputs = []
        for start in range(0, len(row["syms"]), batch_size):
            batch = {k: v[start:start + batch_size].to(device) for k, v in encoded.items()}
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            outputs.append(F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1).cpu())
        docs = torch.cat(outputs)
        for start in range(0, len(docs) - episode_length + 1, episode_length):
            all_embeddings.append(F.normalize(docs[start:start + episode_length].mean(0), dim=0).numpy())
            all_labels.append(row["author_id"])
    return np.asarray(all_embeddings), np.asarray(all_labels)


def run(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    qrows = read_histories(args.data_dir / "test_queries.jsonl")
    trows = read_histories(args.data_dir / "test_targets.jsonl")

    luar, luar_tok, cfg = load_export(args.luar_model, device)
    qcache = tokenize_histories(qrows, luar_tok, args.max_length)
    tcache = tokenize_histories(trows, luar_tok, args.max_length)
    qeps, qlabels, _ = eval_episodes(qrows, qcache, args.episode_length)
    teps, tlabels, _ = eval_episodes(trows, tcache, args.episode_length)
    q = embed(luar, qeps, args.batch_size, device)
    t = embed(luar, teps, args.batch_size, device)
    luar_metrics, _, _ = compute_metrics(q, qlabels, t, tlabels)

    atok = AutoTokenizer.from_pretrained(args.atesit_model)
    amodel = AutoModel.from_pretrained(args.atesit_model).to(device).eval()
    aq, aqlabels = atesi_embeddings(qrows, atok, amodel, args.max_length, args.episode_length, device, args.batch_size)
    at, atlabels = atesi_embeddings(trows, atok, amodel, args.max_length, args.episode_length, device, args.batch_size)
    atesi_metrics, _, _ = compute_metrics(aq, aqlabels, at, atlabels)

    report = {
        "holdout": "test_queries vs test_targets",
        "device": str(device), "episode_length": args.episode_length, "max_length": args.max_length,
        "luar_model": str(args.luar_model), "atesit_model": args.atesit_model,
        "luar": luar_metrics, "atesit": atesi_metrics,
        "delta_luar_minus_atesit": {
            k: luar_metrics[k] - atesi_metrics[k]
            for k in ("roc_auc", "average_precision", "eer", "pair_balanced_accuracy",
                      "pair_f1", "accuracy", "macro_f1", "balanced_accuracy", "recall_at_1", "mrr", "cosine_gap")
            if k in luar_metrics and k in atesi_metrics
        },
        "caveats": [
            "The holdout contains two known channels; these metrics do not establish unseen-channel generalization.",
            "AtesiT is evaluated with mean pooling over post embeddings and the same non-overlapping episode protocol.",
            "Thresholds are independently selected from each model's own holdout scores; use a separate calibration split for production thresholding.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "holdout_comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--luar-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--atesit-model", default="AtesiT/osint-stylometry-model")
    p.add_argument("--episode-length", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
