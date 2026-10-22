"""v18: evaluate new split predictions, review facts, fail closed on inference."""

from __future__ import annotations

import sys

import dual_qlora_inference_v4 as implementation
import dual_qlora_inference_v5 as validation
import dual_qlora_inference_v8 as strict
import news_style_runtime_v18 as runtime


implementation.runtime = runtime
implementation.semantic_review = strict.strict_semantic_review
_check_and_revise = implementation.check_and_revise


def check_and_revise(model, tokenizer, source: str, candidate: str, mask: dict, channel: str):
    result, review = _check_and_revise(model, tokenizer, source, candidate, mask, channel)
    if review["needs_review"]:
        # Neither draft nor revision passed the independent checks. Never label
        # an unverified model completion as publishable: retain it in raw_response
        # and fall back to the exact source text for the displayed response.
        review["fallback_to_source"] = True
        review["unverified_draft"] = result
        return source, review
    review["fallback_to_source"] = False
    return result, review


implementation.check_and_revise = check_and_revise


def main() -> None:
    work, channels = validation.adapter_paths_from_argv(sys.argv[1:])
    implementation.main()
    if "--validation-only" not in sys.argv[1:]:
        validation.validate_outputs(work, channels)
    print("v18 inference validated", len(channels), "channels")


if __name__ == "__main__":
    main()
