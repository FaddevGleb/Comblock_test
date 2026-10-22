"""Generic multi-adapter inference, source-grounded review and versioned reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import news_style_runtime_v14 as runtime


FACT_CONFIDENCE_THRESHOLD = 0.85
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "num_beams": 1,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


class RussianTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return runtime.words(text)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def complete(model, tokenizer, messages: list[dict], max_tokens: int = 256) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=GENERATION_CONFIG["do_sample"],
            num_beams=GENERATION_CONFIG["num_beams"],
            repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def semantic_review(model, tokenizer, source: str, candidate: str) -> dict:
    prompt = [
        {"role": "system", "content": "Проверь фактическую поддержку поста исходной новостью. Верни строго JSON: {\"supported\":true/false,\"confidence\":0.0-1.0,\"unsupported_claims\":[\"...\"]}. Сомнительные проверяемые утверждения считай неподтверждёнными. Метафоры без проверяемого факта не считай ошибкой."},
        {"role": "user", "content": f"ИСТОЧНИК:\n{source}\n\nПОСТ:\n{candidate}\n\nПроверяй участников, действия, числа, место, время, причины, последствия, отрицание и статус: предложение/решение/событие."},
    ]
    with model.disable_adapter():
        answer = complete(model, tokenizer, prompt, max_tokens=180)
    try:
        match = re.search(r"\{.*\}", answer, re.S)
        parsed = json.loads(match.group()) if match else None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("supported"), bool):
            raise ValueError("missing supported")
        confidence = float(parsed.get("confidence", 0))
        claims = parsed.get("unsupported_claims") or []
        if not isinstance(claims, list):
            claims = [str(claims)]
        return {"supported": parsed["supported"], "confidence": confidence, "unsupported_claims": claims, "parse_ok": True}
    except (ValueError, TypeError, AttributeError):
        return {"supported": False, "confidence": 0.0, "unsupported_claims": [], "parse_ok": False, "raw_review": answer[:500]}


def check_and_revise(model, tokenizer, source: str, candidate: str, mask: dict, channel: str) -> tuple[str, dict]:
    deterministic = runtime.factual_check(source, candidate)
    semantic = semantic_review(model, tokenizer, source, candidate)
    style = runtime.style_check(candidate, mask)
    verified = deterministic["deterministic_passed"] and semantic["supported"] and semantic["confidence"] >= FACT_CONFIDENCE_THRESHOLD and style["passed"]
    review = {"initial_deterministic": deterministic, "initial_semantic": semantic, "initial_style": style, "revision_attempted": False}
    if verified:
        review["needs_review"] = False
        return candidate, review

    model.set_adapter(channel)
    revised = complete(model, tokenizer, runtime.revision_messages(source, candidate, {"deterministic": deterministic, "semantic": semantic, "style": style}, mask))
    revised_deterministic = runtime.factual_check(source, revised)
    revised_semantic = semantic_review(model, tokenizer, source, revised)
    revised_style = runtime.style_check(revised, mask)
    verified = revised_deterministic["deterministic_passed"] and revised_semantic["supported"] and revised_semantic["confidence"] >= FACT_CONFIDENCE_THRESHOLD and revised_style["passed"]
    review.update({"revision_attempted": True, "revised_deterministic": revised_deterministic, "revised_semantic": revised_semantic, "revised_style": revised_style, "needs_review": not verified})
    return revised, review


def generate_case(model, tokenizer, source: str, channel: str, mask: dict, *, reference: str | None = None, case_id=None) -> dict:
    model.set_adapter(channel)
    row = {"instruction": "", "input": source, "output": ""}
    prompt, _ = runtime.messages(row, channel, mask)
    raw = complete(model, tokenizer, prompt)
    result, review = check_and_revise(model, tokenizer, source, raw, mask, channel)
    output = {
        "id": case_id, "channel": channel, "source_text": source,
        "raw_response": raw, "response": result, "fact_review": review,
        "style_comparison": runtime.style_comparison(result, mask),
        "length_plan": runtime.length_plan(source, mask),
        "active_adapter": list(model.active_adapters),
    }
    if reference is not None:
        output["reference"] = reference
    return output


def add_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    bleu = BLEU(tokenize="intl")
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=RussianTokenizer(), use_stemmer=False)
    for row in rows:
        score = rouge.score(row["reference"], row["response"])
        row["metrics"] = {
            "bleu": round(bleu.sentence_score(row["response"], [row["reference"]]).score, 4),
            **{key + "_f1": round(score[key].fmeasure, 4) for key in ("rouge1", "rouge2", "rougeL")},
        }
    return {
        "corpus_bleu": round(bleu.corpus_score([r["response"] for r in rows], [[r["reference"] for r in rows]]).score, 4),
        "mean_rougeL_f1": round(sum(r["metrics"]["rougeL_f1"] for r in rows) / len(rows), 4),
        "needs_review_rate": round(sum(r["fact_review"]["needs_review"] for r in rows) / len(rows), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Сохранить только validation_results.jsonl и не запускать test/full generation.",
    )
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    global FACT_CONFIDENCE_THRESHOLD, GENERATION_CONFIG
    policy = config.get("policy", {})
    FACT_CONFIDENCE_THRESHOLD = float(policy.get("semantic_confidence_threshold", FACT_CONFIDENCE_THRESHOLD))
    GENERATION_CONFIG = {**GENERATION_CONFIG, **policy.get("generation", {})}
    channels = [item["name"] for item in config["channels"]]
    masks = {name: runtime.load_mask(args.prepared / f"mask_{name}.json") for name in channels}
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], quantization_config=quant, torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="eager")
    model = PeftModel.from_pretrained(base, args.work / "adapters" / channels[0], adapter_name=channels[0])
    for channel in channels[1:]:
        model.load_adapter(args.work / "adapters" / channel, adapter_name=channel)
    model.eval()
    args.work.mkdir(parents=True, exist_ok=True)
    audit = {"model_id": config["model_id"], "base_instance_count": 1, "adapters": {}}
    source = read_jsonl(args.test)[0].get("text", "")
    prompt, _ = runtime.messages({"input": source, "output": ""}, channels[0], masks[channels[0]])
    encoded = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(encoded, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(**batch).logits[0, -1].float()
        for channel in channels:
            model.set_adapter(channel)
            logits = model(**batch).logits[0, -1].float()
            audit["adapters"][channel] = {"active": list(model.active_adapters), "max_logit_delta_vs_base": float((logits - base_logits).abs().max())}
    (args.work / "adapter_activation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = []
    for channel in channels:
        metrics_dir = args.work / "workers" / channel / "metrics"
        summary = json.loads((metrics_dir / "summary.json").read_text(encoding="utf-8"))
        for item in read_jsonl(metrics_dir / f"split_epoch_{summary['best_epoch']}.jsonl"):
            # These predictions were produced by the split model; never regenerate validation with the final adapter.
            raw = item["prediction"]
            deterministic = runtime.factual_check(item["input"], raw)
            semantic = semantic_review(model, tokenizer, item["input"], raw)
            style = runtime.style_check(raw, masks[channel])
            review = {"initial_deterministic": deterministic, "initial_semantic": semantic, "initial_style": style, "revision_attempted": False, "needs_review": not (deterministic["deterministic_passed"] and semantic["supported"] and semantic["confidence"] >= FACT_CONFIDENCE_THRESHOLD and style["passed"])}
            validation.append({"channel": channel, "source_text": item["input"], "reference": item["reference"], "raw_response": raw, "response": raw, "fact_review": review, "style_comparison": runtime.style_comparison(raw, masks[channel])})
    # The final adapter has seen audited validation pairs, so only split predictions are scored.
    raw_metrics = add_metrics(validation)
    write_jsonl(args.work / "validation_results.jsonl", validation)

    if args.validation_only:
        summary = {
            "validation": raw_metrics,
            "validation_count": len(validation),
            "test_needs_review": None,
            "test_count": 0,
            "mode": "validation_only",
        }
        (args.work / "evaluation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "validation-only inference done",
            len(validation),
            json.dumps(raw_metrics, ensure_ascii=False),
            flush=True,
        )
        return

    test = []
    for item in read_jsonl(args.test):
        source = item.get("text") or item.get("input")
        for channel in channels:
            test.append(generate_case(model, tokenizer, source, channel, masks[channel], case_id=item.get("id")))
    write_jsonl(args.work / "test_predictions.jsonl", test)

    full_dir = args.work / "full_generation"
    full_dir.mkdir(exist_ok=True)
    for channel in channels:
        rows = read_jsonl(args.manifest.parent / next(x["supervised"] for x in config["channels"] if x["name"] == channel))
        full = [generate_case(model, tokenizer, row["input"], channel, masks[channel], reference=row["output"], case_id=i) for i, row in enumerate(rows, 1)]
        add_metrics(full)
        write_jsonl(full_dir / f"{channel}_full_generation.jsonl", full)
        lines = [f"# Полная генерация — {channel}", "", "Все примеры были доступны при финальном обучении, если не отклонены аудитом. Эти результаты не являются отложенной валидацией.", ""]
        for item in full:
            lines += [f"## Пример {item['id']}", "", "### Новость", "", item["source_text"], "", "### Эталон", "", item["reference"], "", "### Генерация", "", item["response"], "", "### Проверка", "", f"- needs_review: `{item['fact_review']['needs_review']}`", f"- BLEU: `{item['metrics']['bleu']}`; ROUGE-L: `{item['metrics']['rougeL_f1']}`", ""]
        (full_dir / f"{channel}_full_generation.md").write_text("\n".join(lines), encoding="utf-8")

    (args.work / "evaluation_summary.json").write_text(json.dumps({"validation": raw_metrics, "test_needs_review": sum(x["fact_review"]["needs_review"] for x in test), "test_count": len(test)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("inference done", len(validation), len(test), json.dumps(raw_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
