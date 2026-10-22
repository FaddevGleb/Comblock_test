"""Verify adapter activation and generate every requested channel variant."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import style_runtime_v3 as runtime


MODEL_ID = runtime.MODEL_ID
WORK = Path("/kaggle/working/dual_qlora")
TEST_INPUTS = Path("/kaggle/working/dual_qlora_input_v4/test_inputs.jsonl")
CHANNELS = ("banki_oil", "moscowach")
# Task-label mapping, not a style description or prompt instruction.
CHANNEL_TYPES = {"banki_oil": "type1", "moscowach": "type2"}


class RussianTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return runtime.WORDS.findall(text.casefold())


def active_names(model) -> list[str]:
    """PEFT exposes active_adapters as a property; accept callable variants too."""
    value = model.active_adapters
    return list(value() if callable(value) else value)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    masks = {channel: runtime.load_mask(WORK / "adapters" / channel / "style_mask.json") for channel in CHANNELS}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=quant, torch_dtype=torch.float16,
        device_map={"": 0}, attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, WORK / "adapters" / "banki_oil", adapter_name="banki_oil")
    model.load_adapter(WORK / "adapters" / "moscowach", adapter_name="moscowach")
    model.eval()

    test_rows = [
        json.loads(line) for line in TEST_INPUTS.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    probe_row = {"instruction": "Напиши один пост для канала.", "input": test_rows[0]["text"], "output": ""}
    prompt, _ = runtime.messages(probe_row, "banki_oil", masks["banki_oil"])
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    audit = {
        "model_id": MODEL_ID,
        "base_instance_count": 1,
        "loaded_adapters": sorted(model.peft_config.keys()),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    }
    with torch.inference_mode():
        with model.disable_adapter():
            base_logits = model(**batch).logits[0, -1].float()
        for channel in CHANNELS:
            model.set_adapter(channel)
            assert channel in active_names(model), f"{channel} is not active"
            adapted_logits = model(**batch).logits[0, -1].float()
            difference = float((adapted_logits - base_logits).abs().max())
            if difference <= 1e-6:
                raise RuntimeError(f"{channel}: adapter has no measurable effect on logits")
            audit[channel] = {"active_adapters": active_names(model), "max_logit_delta_vs_base": difference}
    (WORK / "adapter_activation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test_results = []
    for row in test_rows:
        for channel in CHANNELS:
            model.set_adapter(channel)
            assert channel in active_names(model)
            request = {"instruction": "Напиши один пост для канала.", "input": row["text"], "output": ""}
            messages, _ = runtime.messages(request, channel, masks[channel])
            encoded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(encoded, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False, num_beams=1,
                    repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id, use_cache=True,
                )
            raw_output = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            output, guard = runtime.guard_output(row["text"], raw_output, channel, masks[channel])
            test_results.append({
                "kind": "test", "id": row["id"], "channel": channel,
                "type": CHANNEL_TYPES[channel],
                "source_text": row["text"], "response": output, "raw_response": raw_output,
                "factual_guard": guard, "length_plan": runtime.length_plan(row["text"], masks[channel]),
                "style_comparison": runtime.style_comparison(output, masks[channel]),
                "active_adapter": active_names(model),
                "prompt_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            })
    write_jsonl(WORK / "test_predictions.jsonl", test_results)

    bleu = BLEU(tokenize="intl")
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=RussianTokenizer(), use_stemmer=False)
    validation_results = []
    for channel in CHANNELS:
        root = WORK / "workers" / channel / "metrics"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        path = root / f"split_epoch_{summary['best_epoch']}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            scores = rouge.score(item["reference"], item["prediction"])
            validation_results.append({
                "kind": "validation", "channel": channel,
                "type": CHANNEL_TYPES[channel],
                "source_text": item["input"], "reference": item["reference"],
                "response": item["prediction"],
                "factual_guard": item.get("factual_check", runtime.factual_check(item["input"], item["prediction"])),
                "length_plan": item.get("length_plan", runtime.length_plan(item["input"], masks[channel])),
                "style_comparison": runtime.style_comparison(item["prediction"], masks[channel]),
                "metrics": {
                    "bleu": round(bleu.sentence_score(item["prediction"], [item["reference"]]).score, 6),
                    **{name + "_f1": round(scores[name].fmeasure, 6) for name in ("rouge1", "rouge2", "rougeL")},
                },
            })
    write_jsonl(WORK / "validation_results.jsonl", validation_results)
    write_jsonl(WORK / "final_results.jsonl", validation_results + test_results)
    print("adapter activation verified:", audit)
    print("validation:", len(validation_results), "test variants:", len(test_results))


if __name__ == "__main__":
    main()
