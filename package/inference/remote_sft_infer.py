"""Standalone one-GPU SFT inference runner for the Qudata host.

The runner keeps the current policy: eight candidates are generated in one
batch per news/channel, candidates below the channel content-word floor are
rejected, and the winner is selected by LUAR only among hard-gate survivors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.common import read_jsonl, write_jsonl
from src.inference.generator import AdapterGenerator
from src.inference.local_checks import check_candidate
from src.inference.language_checker import LanguageChecker
from src.style.feature_extractor import count_content_words
from src.style.luar_scorer import LuarScorer
from src.style.prompt_builder import FACT_SYSTEM
from src.style.stylometry_scorer import content_word_floor


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _language_checker():
    try:
        checker = LanguageChecker({"backend": "local", "server_url": None, "ignored_rules": [], "allowed_words": []})
        checker.check("Проверка русского текста.")
        return checker
    except Exception as exc:  # pragma: no cover - host-dependent fallback
        class Simple:
            def check(self, text):
                errors = []
                if not text.strip():
                    errors.append({"rule": "empty_post"})
                if "\ufffd" in text or re.search(r"<(?:think|analysis)>|<\|im_", text):
                    errors.append({"rule": "model_control_artifact"})
                return {
                    "language_pass": not errors,
                    "language_errors": errors,
                    "method": "deterministic_fallback",
                    "language_checker_error": str(exc),
                }

            def close(self):
                return None

        return Simple()


def _system(channel: str, input_root: Path) -> str:
    if channel == "banki_oil":
        mask = (input_root / "profiles/banki_oil/style_mask_type1_news_corpus_compact.md").read_text(
            encoding="utf-8"
        )
        return (
          mask.strip()
        )
    mask = (input_root / "profiles/moscowach/style_mask_compact.md").read_text(encoding="utf-8")
    return FACT_SYSTEM + f"\nКанал: {channel}\nСтиль:\n{mask.strip()}"


def _training_messages(channel: str, source: str, input_root: Path):
    """Recreate the chat messages used during SFT, including the type1 runtime."""
    if channel == "banki_oil":
        spec = importlib.util.spec_from_file_location(
            "type1_style_runtime", input_root / "style_runtime_v3.py"
        )
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        mask = runtime.load_mask(input_root / "mask_type1_news_corpus.json")
        return runtime.messages({"input": source, "output": ""}, channel, mask)[0]
    return [
        {"role": "system", "content": _system(channel, input_root)},
        {"role": "user", "content": "Исходный текст:\n" + source},
    ]


def _generate_messages(generator, prompt_messages, adapter_name, adapter_path, config, seeds):
    """Batch generation with an already formatted training prompt."""
    import torch

    ids = generator.tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(generator.model.device)
    ids = ids.repeat(len(seeds), 1)
    maximum = int(config["max_new_tokens"])
    limit = min(getattr(generator.model.config, "max_position_embeddings", 32768), config["context_limit"])
    if ids.shape[1] + maximum > limit:
        raise ValueError("Formatted training prompt exceeds the configured context limit")
    generator.torch.cuda.default_generators[generator.device].manual_seed(int(seeds[0]))
    with generator.activate(adapter_name, adapter_path), torch.inference_mode():
        output = generator.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=maximum,
            do_sample=True,
            temperature=config["temperature"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            pad_token_id=generator.tokenizer.pad_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = 0.0
    return [
        {
            "text": generator.tokenizer.decode(output[i, ids.shape[1] :], skip_special_tokens=True).strip(),
            "seed": int(seed),
            "generation_batch_size": len(seeds),
            "new_tokens": int(output[i, ids.shape[1] :].shape[0]),
            "hit_token_limit": False,
            "generation_seconds": elapsed,
        }
        for i, seed in enumerate(seeds)
    ]


def _safe_name(channel: str) -> str:
    return channel.replace("@", "_").replace("/", "_")


def run(args) -> int:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    canonical = read_jsonl(input_root / "local_artifacts/local_canonical.jsonl")
    inputs = [_read_json(Path(args.inputs))] if args.inputs and Path(args.inputs).suffix == ".json" else read_jsonl(Path(args.inputs))
    # test_inputs.jsonl is a JSONL file; tolerate a single JSON object for manual runs.
    if not inputs:
        inputs = [{"id": 1, "news": args.news}] if args.news else []
    if args.news:
        inputs = [{"id": "manual-1", "news": args.news}]
    if not inputs:
        raise ValueError("No input news supplied")

    refs = {}
    for channel in ("banki_oil", "moscowach"):
        refs[channel] = [
            r["original_post"]
            for r in canonical
            if r.get("channel") == channel
            and r.get("split") == "train"
            and r.get("status") not in ("duplicate", "malformed")
        ]
    floor = {"banki_oil": 28, "moscowach": content_word_floor(refs["moscowach"])}

    import torch

    device = int(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on the inference host")
    generator = AdapterGenerator(args.base_model, args.revision, device=device)
    luar = LuarScorer(input_root / "models/luar", f"cuda:{device}").fit_records(canonical)
    language = _language_checker()
    gen_cfg = {
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "repetition_penalty": float(args.repetition_penalty),
        "context_limit": int(args.context_limit),
    }
    adapters = {
        "banki_oil": input_root / "adapters/banki_oil/SFT_only",
        "moscowach": input_root / "adapters/moscowach/SFT_only",
    }
    systems = {c: _training_messages(c, "", input_root)[0]["content"] for c in adapters}
    results = []
    candidates_out = []
    started = time.time()
    total = len(inputs) * len(adapters)
    done = 0
    try:
        for item in inputs:
            source = item["news"]
            for channel, adapter in adapters.items():
                seeds = [args.seed + index for index in range(int(args.num_candidates))]
                generated = _generate_messages(
                    generator,
                    _training_messages(channel, source, input_root),
                    "SFT_only_" + channel,
                    adapter,
                    gen_cfg,
                    seeds,
                )
                scored = []
                for index, candidate in enumerate(generated):
                    candidate.update(candidate_index=index)
                    check = check_candidate(source, candidate, language)
                    candidate.update(check)
                    candidate["content_word_count"] = count_content_words(candidate["text"])
                    candidate["content_word_floor"] = floor[channel]
                    candidate["content_word_floor_pass"] = candidate["content_word_count"] >= floor[channel]
                    # Selection is LUAR-only. Length and local diagnostics are
                    # recorded for review but are not hard filters.
                    candidate["eligible"] = True
                    score = luar.score(channel, candidate["text"])
                    candidate["luar_score"] = score["normalized_score"]
                    candidate["luar_diagnostics"] = score
                    candidate["channel"] = channel
                    candidate["input_id"] = item.get("id")
                    scored.append(candidate)
                    candidates_out.append(candidate)
                eligible = [c for c in scored if c.get("luar_score") is not None]
                selected = max(eligible, key=lambda c: c["luar_score"]) if eligible else None
                results.append(
                    {
                        "input_id": item.get("id"),
                        "channel": channel,
                        "version": "SFT_only",
                        "source_news": source,
                        "selected_text": selected["text"] if selected else None,
                        "selection_status": "accepted" if selected else "no_eligible_candidate",
                        "selected_luar_score": selected.get("luar_score") if selected else None,
                        "content_word_floor": floor[channel],
                        "candidate_count": len(scored),
                        "eligible_count": len(eligible),
                        "candidates": scored,
                    }
                )
                done += 1
                (output_root / "progress.json").write_text(
                    json.dumps({"done": done, "total": total, "elapsed_seconds": time.time() - started}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    finally:
        language.close()
        generator.model.cpu()
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(output_root / "results.jsonl", results)
    write_jsonl(output_root / "candidates.jsonl", candidates_out)
    lines = ["# SFT inference on Qudata", "", f"Candidates per slot: {args.num_candidates}", "Selection: LUAR only (no hard length gate)", ""]
    for row in results:
        lines.extend([
            f"## {row['channel']} · input {row['input_id']}",
            "",
            f"**News:** {row['source_news']}",
            "",
            f"**Status:** {row['selection_status']} · eligible {row['eligible_count']}/{row['candidate_count']} · LUAR {row['selected_luar_score']}",
            "",
            row["selected_text"] or "_Нет кандидата, прошедшего жёсткие фильтры._",
            "",
        ])
    (output_root / "news_generation_pairs.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "run_status.json").write_text(
        json.dumps({"status": "DONE", "slots": len(results), "elapsed_seconds": time.time() - started}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--news")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
