"""v16 inference: reuse trained adapters, regenerate with factuality-only prompts."""

from __future__ import annotations

import json
import re
import sys

import dual_qlora_inference_v4 as implementation
import dual_qlora_inference_v5 as previous
import news_style_runtime_v16 as runtime

implementation.runtime = runtime


def strict_semantic_review(model, tokenizer, source: str, candidate: str) -> dict:
    prompt = [
        {
            "role": "system",
            "content": (
                "Проверь пост только на поддержку исходником. Верни строго JSON: "
                '{"supported":true/false,"confidence":0.0-1.0,"unsupported_claims":[]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"SOURCE_NEWS:\n{source}\n\nPOST:\n{candidate}\n\n"
                "Считай неподтверждёнными любые новые числа, имена, места, даты, причины, "
                "последствия, сравнения и конкретные обстоятельства. Если есть сомнение — false."
            ),
        },
    ]
    with model.disable_adapter():
        answer = implementation.complete(model, tokenizer, prompt, max_tokens=180)
    try:
        match = re.search(r"\{.*\}", answer, re.S)
        parsed = json.loads(match.group()) if match else None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("supported"), bool):
            raise ValueError("invalid review")
        claims = parsed.get("unsupported_claims") or []
        return {
            "supported": parsed["supported"],
            "confidence": float(parsed.get("confidence", 0)),
            "unsupported_claims": claims if isinstance(claims, list) else [str(claims)],
            "parse_ok": True,
        }
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return {"supported": False, "confidence": 0.0, "unsupported_claims": [], "parse_ok": False, "raw_review": answer[:500]}


implementation.semantic_review = strict_semantic_review


def main() -> None:
    work, channels = previous.adapter_paths_from_argv(sys.argv[1:])
    implementation.main()
    previous.validate_outputs(work, channels)
    print("v16 factuality-only regeneration complete")


if __name__ == "__main__":
    main()
