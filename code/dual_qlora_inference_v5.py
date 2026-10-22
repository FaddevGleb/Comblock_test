"""v14 inference wrapper with artifact and report completeness checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dual_qlora_inference_v4 as implementation


def adapter_paths_from_argv(argv: list[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    import json
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    return args.work, [entry["name"] for entry in config["channels"]]


def validate_outputs(work: Path, channels: list[str]) -> None:
    required = [work / "validation_results.jsonl", work / "test_predictions.jsonl", work / "evaluation_summary.json"]
    required.extend(work / "full_generation" / f"{channel}_full_generation.jsonl" for channel in channels)
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    empty_adapters = [str(path) for channel in channels for path in [work / "adapters" / channel / "adapter_model.safetensors"] if not path.exists() or path.stat().st_size == 0]
    if empty_adapters:
        raise RuntimeError("Empty adapter artifacts after inference: " + ", ".join(empty_adapters))
    if missing:
        raise RuntimeError("Missing or empty inference artifacts: " + ", ".join(missing))


def main() -> None:
    work, channels = adapter_paths_from_argv(sys.argv[1:])
    implementation.main()
    validate_outputs(work, channels)
    print("v14 inference artifacts validated", len(channels), "channels")


if __name__ == "__main__":
    main()
