"""Train/evaluate LUAR channel embeddings on a single Kaggle GPU.

Default: fine-tune official pretrained LUAR-MUD. --backbone builds the same
mean-token / self-attention / max-episode / linear architecture on another encoder.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

LOG = logging.getLogger("luar")


class ChannelLUAR(nn.Module):
    """LUAR aggregation matching LUAR/src/models/transformer.py."""

    def __init__(self, transformer, embedding_dim=512):
        super().__init__()
        self.transformer = transformer
        self.linear = nn.Linear(transformer.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        batch, episode, length = input_ids.shape
        mask = attention_mask.reshape(-1, length)
        hidden = self.transformer(
            input_ids=input_ids.reshape(-1, length), attention_mask=mask,
            return_dict=True,
        ).last_hidden_state
        # Pool in FP32, also under mixed precision.
        w = mask.unsqueeze(-1).float()
        docs = ((hidden.float() * w).sum(1) / w.sum(1).clamp_min(1)).reshape(batch, episode, -1)
        attention = (docs @ docs.transpose(1, 2) / math.sqrt(docs.shape[-1])).softmax(-1)
        history = (attention @ docs).amax(dim=1)
        return self.linear(history)


def load_model(args):
    if args.backbone:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone)
        model = ChannelLUAR(AutoModel.from_pretrained(args.backbone), args.embedding_dim)
        kind = "multilingual_luar_architecture"
    else:
        # LUAR's HF architecture lives in the upstream repo's config.py/model.py.
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        if model.config.model_type.lower() != "luar":
            raise ValueError("--model must be a pretrained LUAR model. For a plain encoder use --backbone.")
        kind = "pretrained_luar"
    return model, tokenizer, kind


def save_model(model, tokenizer, path: Path, args, kind):
    path.mkdir(parents=True, exist_ok=True)
    if kind == "pretrained_luar":
        model.save_pretrained(path, safe_serialization=True)
    else:
        model.transformer.save_pretrained(path / "backbone", safe_serialization=True)
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in model.linear.state_dict().items()},
                  str(path / "projection.safetensors"))
    tokenizer.save_pretrained(path)
    write_json(path / "channel_luar_config.json", {
        "kind": kind, "embedding_dim": model.linear.out_features,
        "max_length": args.max_length, "episode_length": args.episode_length,
        "normalize_embeddings": True,
        "aggregation": "mean_tokens -> dot_self_attention -> max_posts -> linear",
        "source": args.backbone or args.model,
    })


def load_export(path: Path, device="cpu"):
    """Load best_model for inference, without access to training data."""
    cfg = json.loads((path / "channel_luar_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    if cfg["kind"] == "pretrained_luar":
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
    else:
        from safetensors.torch import load_file
        model = ChannelLUAR(AutoModel.from_pretrained(path / "backbone"), cfg["embedding_dim"])
        model.linear.load_state_dict(load_file(str(path / "projection.safetensors")))
    return model.to(device).eval(), tokenizer, cfg


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_histories(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = []
    for r in rows:
        if not isinstance(r.get("author_id"), int) or not isinstance(r.get("syms"), list):
            raise ValueError(f"{path}: expected integer author_id and list syms")
        if not r["syms"] or not all(isinstance(t, str) and t.strip() for t in r["syms"]):
            raise ValueError(f"{path}: empty/invalid texts")
        ids.append(r["author_id"])
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError(f"{path}: expected >=2 distinct channels, one row per channel")
    return rows


def validate_splits(data):
    """Reject exact clean-text leakage even when the converter was bypassed."""
    owners = {}
    for name, rows in data.items():
        for row in rows:
            for text in row["syms"]:
                key = " ".join(text.casefold().split())
                if key in owners:
                    raise ValueError(f"Duplicate text in {name} and {owners[key]}")
                owners[key] = name
    for split in ("validation", "test"):
        q = {r["author_id"] for r in data[split + "_queries"]}
        t = {r["author_id"] for r in data[split + "_targets"]}
        if q != t:
            raise ValueError(f"{split}: query/gallery channel IDs must match")


def tokenize_histories(rows, tokenizer, max_length):
    return {
        r["author_id"]: tokenizer(r["syms"], padding="max_length", truncation=True,
                                 max_length=max_length, return_tensors="pt")
        for r in rows
    }


def supcon_loss(embeddings, labels, temperature):
    """Supervised contrastive loss: positives share channel, self pairs excluded."""
    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.T / temperature
    self_mask = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    if not positives.any(dim=1).all() or labels.unique().numel() < 2:
        raise ValueError("Every channel needs >=2 episodes; batch needs >=2 channels")
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probs.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()


def sample_batch(cache, rng, channels_per_batch, views, episode_length, device):
    channels = rng.sample(sorted(cache), channels_per_batch)
    batches = {"input_ids": [], "attention_mask": []}
    labels = []
    for channel in channels:
        # No shared post between any positive episodes in a training batch.
        count = views * episode_length
        n = len(cache[channel]["input_ids"])
        if n < count:
            raise ValueError(f"Channel {channel}: need {count} train posts, have {n}")
        indices = torch.tensor(rng.sample(range(n), count))
        for key in batches:
            batches[key].append(cache[channel][key][indices].reshape(views, episode_length, -1))
        labels.extend([channel] * views)
    return {k: torch.cat(v).to(device) for k, v in batches.items()}, torch.tensor(labels, device=device)


def eval_episodes(rows, cache, episode_length):
    episodes, labels, post_ids = [], [], []
    for row in rows:
        channel = row["author_id"]
        n = len(row["syms"])
        # Fixed, disjoint episodes. Never repeat posts to pad the last episode.
        for start in range(0, n - episode_length + 1, episode_length):
            episodes.append({k: cache[channel][k][start:start + episode_length]
                             for k in ("input_ids", "attention_mask")})
            labels.append(channel)
            post_ids.append(row.get("post_ids", list(range(n)))[start:start + episode_length])
    if set(labels) != {r["author_id"] for r in rows}:
        raise ValueError("Not enough evaluation posts for the requested episode_length")
    return episodes, np.asarray(labels), post_ids


@torch.no_grad()
def embed(model, episodes, batch_size, device):
    model.eval()
    output = []
    for start in range(0, len(episodes), batch_size):
        batch = {k: torch.stack([e[k] for e in episodes[start:start + batch_size]]).to(device)
                 for k in ("input_ids", "attention_mask")}
        output.append(F.normalize(model(**batch).float(), dim=-1).cpu().numpy())
    return np.concatenate(output)


def verification_metrics(labels, scores, threshold=None):
    y = np.asarray(labels).ravel().astype(int)
    s = np.asarray(scores).ravel()
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Verification requires both same-channel and different-channel pairs")
    fpr, tpr, thresholds = roc_curve(y, s)
    j = int(np.argmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[j] + 1 - tpr[j]) / 2)
    # Approximate EER operating point, calibrated ONLY on validation.
    if threshold is None:
        threshold = float(thresholds[j])
        if not math.isfinite(threshold):
            threshold = float(np.max(s) + 1e-6)
    pred = s >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "eer": eer, "threshold": threshold,
        "pair_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "pair_f1": float(f1_score(y, pred, zero_division=0)),
        "same_cosine": float(s[y == 1].mean()), "different_cosine": float(s[y == 0].mean()),
        "cosine_gap": float(s[y == 1].mean() - s[y == 0].mean()),
        "positive_pairs": int(y.sum()), "negative_pairs": int((y == 0).sum()),
    }


def compute_metrics(query, query_labels, gallery, gallery_labels, threshold=None):
    classes = np.unique(gallery_labels)
    prototypes = np.stack([gallery[gallery_labels == c].mean(0) for c in classes])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    scores = query @ prototypes.T
    order = np.argsort(-scores, axis=1, kind="stable")
    prediction = classes[order[:, 0]]
    ranks = np.argmax(classes[order] == query_labels[:, None], axis=1) + 1
    pair_scores = query @ gallery.T
    pair_labels = query_labels[:, None] == gallery_labels[None, :]
    metrics = verification_metrics(pair_labels, pair_scores, threshold)
    metrics.update({
        "accuracy": float(accuracy_score(query_labels, prediction)),
        "macro_f1": float(f1_score(query_labels, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(query_labels, prediction)),
        "recall_at_1": float(np.mean(ranks == 1)), "mrr": float(np.mean(1 / ranks)),
        "chance_accuracy": float(1 / len(classes)),
        "channels": len(classes), "query_episodes": len(query), "gallery_episodes": len(gallery),
        "confusion_matrix": confusion_matrix(query_labels, prediction, labels=classes).tolist(),
        "class_ids": classes.tolist(),
    })
    # R@5 with two channels would trivially be 1; don't report it.
    for k in (5, 10):
        if k < len(classes):
            metrics[f"recall_at_{k}"] = float(np.mean(ranks <= k))
    return metrics, pair_scores, pair_labels


def bootstrap_auc(pair_scores, pair_labels, query_labels, seed, repeats):
    """Query-cluster bootstrap; gallery is fixed, so CI is conditional on it."""
    rng = np.random.default_rng(seed)
    samples = []
    groups = [np.flatnonzero(query_labels == c) for c in np.unique(query_labels)]
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        samples.append(roc_auc_score(pair_labels[ix].ravel(), pair_scores[ix].ravel()))
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])] if samples else None


def evaluate(model, plans, args, device, split, thresholds=None, bootstrap=False):
    report, artifacts = {}, {}
    for length in sorted({1, args.episode_length}):
        qepisodes, qlabels, qids = plans[(split, "queries", length)]
        tepisodes, tlabels, tids = plans[(split, "targets", length)]
        q = embed(model, qepisodes, args.eval_batch_size, device)
        t = embed(model, tepisodes, args.eval_batch_size, device)
        key = f"episode_{length}"
        threshold = thresholds[key]["threshold"] if thresholds else None
        m, scores, labels = compute_metrics(q, qlabels, t, tlabels, threshold)
        if bootstrap:
            m["roc_auc_ci95_conditional_gallery"] = bootstrap_auc(scores, labels, qlabels, args.seed, args.bootstrap)
        report[key] = m
        artifacts[key] = {"query_embeddings": q, "query_labels": qlabels,
                          "gallery_embeddings": t, "gallery_labels": tlabels,
                          "query_post_ids": qids, "gallery_post_ids": tids}
    return report, artifacts


def configure_trainable(model, last_layers):
    if last_layers == -1:
        return
    for p in model.transformer.parameters():
        p.requires_grad = False
    layers = model.transformer.encoder.layer
    if last_layers > len(layers):
        raise ValueError(f"--train-last-layers exceeds {len(layers)} transformer layers")
    for layer in list(layers)[len(layers) - last_layers:] if last_layers else []:
        for p in layer.parameters():
            p.requires_grad = True


def run(args):
    if args.epochs < 1 or args.steps_per_epoch < 0 or args.episode_length < 1 or args.views < 2:
        raise ValueError("Invalid epochs, steps, episode length or views")
    if args.train_last_layers < -1 or args.eval_batch_size < 1 or args.patience < 1:
        raise ValueError("Invalid layer count, eval batch size or patience")
    if args.temperature <= 0 or min(args.lr, args.head_lr) <= 0:
        raise ValueError("Temperature and learning rates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Use an empty --output-dir; current directory contains files: {args.output_dir}")
    names = ["train", "validation_queries", "validation_targets", "test_queries", "test_targets"]
    data = {name: read_histories(args.data_dir / (name + ".jsonl")) for name in names}
    validate_splits(data)
    train_channels = len(data["train"])
    cpb = min(args.channels_per_batch, train_channels)
    if cpb < 2:
        raise ValueError("Batch must contain >=2 channels")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "train.log", encoding="utf-8")],
                        force=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    LOG.info("Device=%s train_channels=%s; known-channel and topic effects must be considered.", device, train_channels)
    if train_channels < 10:
        LOG.warning("Only %d train channels: this run cannot establish universal style generalization.", train_channels)
    model, tokenizer, kind = load_model(args)
    configure_trainable(model, args.train_last_layers)
    if args.gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    caches = {name: tokenize_histories(rows, tokenizer, args.max_length) for name, rows in data.items()}
    plans = {}
    for split in ("validation", "test"):
        for side in ("queries", "targets"):
            name = f"{split}_{side}"
            for length in sorted({1, args.episode_length}):
                plans[(split, side, length)] = eval_episodes(data[name], caches[name], length)
    for c, items in caches["train"].items():
        if len(items["input_ids"]) < args.views * args.episode_length:
            raise ValueError(f"Channel {c}: insufficient posts for disjoint positive episodes")
    count = sum(len(r["syms"]) for r in data["train"])
    steps = args.steps_per_epoch or max(1, math.ceil(count / (cpb * args.views * args.episode_length)))
    backbone = [p for p in model.transformer.parameters() if p.requires_grad]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("transformer.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                                  weight_decay=args.weight_decay)
    total_steps = steps * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    use_amp = device.type == "cuda" and args.precision != "fp32"
    bf16 = use_amp and args.precision in ("auto", "bf16") and torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not bf16:
        raise ValueError("BF16 requested but not available on this device")
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not bf16)
    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output_dir / "tensorboard"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update({"kind": kind, "trainable_parameters": trainable, "device_used": str(device),
                          "steps_per_epoch_used": steps, "channels_per_batch_used": cpb,
                          "torch_version": torch.__version__,
                          "model_resolved_commit": getattr(getattr(model, "config", None), "_commit_hash", None),
                          "dataset_hashes": {n: hashlib.sha256((args.data_dir / (n + ".jsonl")).read_bytes()).hexdigest() for n in names}})
    write_json(args.output_dir / "run_config.json", configuration)
    LOG.info("Trainable=%s steps/epoch=%s episodes/batch=%s max_tokens/post=%s", trainable, steps, cpb * args.views, args.max_length)
    baseline_val, _ = evaluate(model, plans, args, device, "validation")
    baseline_test, _ = evaluate(model, plans, args, device, "test", baseline_val, bootstrap=True)
    write_json(args.output_dir / "baseline_metrics.json", {"validation": baseline_val, "test": baseline_test})
    LOG.info("Baseline validation: %s", json.dumps(baseline_val))
    primary = f"episode_{args.episode_length}"
    def selection(metrics):
        return (metrics[primary]["roc_auc"] + metrics[primary]["macro_f1"]) / 2
    best_score, best_epoch = selection(baseline_val), 0
    best_val = baseline_val
    best_path = args.output_dir / "best_state.pt"
    torch.save(model.state_dict(), best_path)
    rng = random.Random(args.seed)
    history_path = args.output_dir / "metrics.jsonl"
    global_step, stale = 0, 0
    with (args.output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as csvfile:
        fields = ["step", "epoch", "loss", "lr", "head_lr", "grad_norm", "step_seconds", "gpu_peak_mb"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=fields)
        csvwriter.writeheader()
        for epoch in range(1, args.epochs + 1):
            losses = []
            for _ in range(steps):
                start = time.perf_counter()
                model.train()
                batch, labels = sample_batch(caches["train"], rng, cpb, args.views, args.episode_length, device)
                optimizer.zero_grad(set_to_none=True)
                current_lrs = [g["lr"] for g in optimizer.param_groups]
                context = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with context:
                    loss = supcon_loss(model(**batch), labels, args.temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; try --precision fp32 or a lower learning rate")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                global_step += 1
                loss_value = float(loss.detach())
                losses.append(loss_value)
                row = dict(zip(fields, [global_step, epoch, loss_value, *current_lrs, float(grad_norm),
                                        time.perf_counter() - start,
                                        torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0]))
                csvwriter.writerow(row)
                csvfile.flush()
                if global_step % args.log_every == 0 or global_step == 1:
                    LOG.info("step=%s epoch=%s loss=%.5f lr=%.2g grad=%.3f seconds=%.2f GPU_peak_MB=%.1f",
                             global_step, epoch, loss_value, row["lr"], row["grad_norm"], row["step_seconds"], row["gpu_peak_mb"])
                if writer:
                    for k in ("loss", "lr", "head_lr", "grad_norm", "gpu_peak_mb"):
                        writer.add_scalar("train/" + k, row[k], global_step)
            val, _ = evaluate(model, plans, args, device, "validation")
            score = selection(val)
            improved = score > best_score + args.min_delta
            if improved:
                best_score, best_epoch, best_val, stale = score, epoch, val, 0
                torch.save(model.state_dict(), best_path)
            else:
                stale += 1
            record = {"epoch": epoch, "step": global_step, "train_loss": float(np.mean(losses)),
                      "validation": val, "selection_score": score, "is_best": improved}
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if writer:
                for key, metrics in val.items():
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"validation/{key}/{name}", value, global_step)
                writer.flush()
            LOG.info("epoch=%s loss=%.5f validation=%s best_epoch=%s", epoch, np.mean(losses), json.dumps(val), best_epoch)
            # Last checkpoint includes optimizer/scheduler/scaler for subsequent manual recovery.
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "step": global_step, "config": configuration}, args.output_dir / "last_checkpoint.pt")
            if stale >= args.patience:
                LOG.info("Early stopping: no validation improvement for %d epochs", stale)
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_test, artifacts = evaluate(model, plans, args, device, "test", best_val, bootstrap=True)
    delta = {key: {m: final_test[key][m] - baseline_test[key][m]
                   for m in ("roc_auc", "eer", "macro_f1", "recall_at_1", "mrr", "cosine_gap")}
             for key in final_test}
    report = {"best_epoch": best_epoch, "selection": "mean(validation ROC-AUC, validation macro-F1)",
              "baseline_validation": baseline_val, "best_validation": best_val,
              "baseline_test": baseline_test, "best_test": final_test, "test_delta": delta,
              "notes": ["Thresholds calibrated on validation and fixed on test.",
                        "best_epoch=0 means fine-tuning did not beat the initial model on validation.",
                        "Bootstrap CI resamples queries, conditional on fixed gallery; not a channel-level generalization CI.",
                        "Channel/content separation is not a proof of topic-independent style."]}
    write_json(args.output_dir / "comparison.json", report)
    for key, artifact in artifacts.items():
        arrays = {k: v for k, v in artifact.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(args.output_dir / ("test_" + key + ".npz"), **arrays)
        write_json(args.output_dir / ("test_" + key + "_post_ids.json"), {k: v for k, v in artifact.items() if k.endswith("post_ids")})
    save_model(model, tokenizer, args.output_dir / "best_model", args, kind)
    write_json(args.output_dir / "best_model" / "thresholds.json", {k: v["threshold"] for k, v in best_val.items()})
    if writer:
        writer.close()
    LOG.info("Done. Best epoch=%s Test deltas=%s Artifacts=%s", best_epoch, json.dumps(delta), args.output_dir)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/luar_run"))
    p.add_argument("--model", default="rrivera1849/LUAR-MUD")
    p.add_argument("--revision", default=None, help="HF model commit, recorded in run config")
    p.add_argument("--backbone", default=None, help="Plain multilingual encoder; creates a NEW LUAR projection")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="0: computed from corpus size")
    p.add_argument("--channels-per-batch", type=int, default=8)
    p.add_argument("--views", type=int, default=4, help="Non-overlapping episodes per channel in each batch")
    p.add_argument("--episode-length", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--train-last-layers", type=int, default=2, help="-1: full fine-tuning; 0: head only")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--no-tensorboard", action="store_true")
    return p


if __name__ == "__main__":
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, FileNotFoundError) as exc:
        cli.error(str(exc))
