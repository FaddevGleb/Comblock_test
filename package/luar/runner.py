"""Kaggle entry point for LUAR-MUD fine-tuning and AtesiT holdout comparison."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUN = ROOT / "run"


def call(*args):
    print("$", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


if __name__ == "__main__":
    call(
        ROOT / "train_kaggle.py",
        "--data-dir", DATA,
        "--output-dir", RUN,
        "--model", "rrivera1849/LUAR-MUD",
        "--epochs", "15",
        "--channels-per-batch", "2",
        "--views", "4",
        "--episode-length", "4",
        "--max-length", "128",
        "--eval-batch-size", "8",
        "--train-last-layers", "2",
        "--lr", "1e-5",
        "--head-lr", "1e-4",
        "--temperature", "0.07",
        "--precision", "auto",
        "--gradient-checkpointing",
        "--patience", "4",
        "--bootstrap", "1000",
        "--seed", "42",
    )
    call(
        ROOT / "compare_holdout.py",
        "--data-dir", DATA,
        "--luar-model", RUN / "best_model",
        "--output-dir", RUN,
        "--episode-length", "4",
        "--max-length", "128",
        "--batch-size", "32",
        "--device", "auto",
    )
