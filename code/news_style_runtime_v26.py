"""v26 runtime: v18 factual contract plus data-driven few-shot style examples."""

from __future__ import annotations

import news_style_runtime_v18 as previous

MODEL_ID = previous.MODEL_ID
words = previous.words
load_mask = previous.load_mask
fold_mask = previous.fold_mask
blind_style_score = previous.blind_style_score
style_comparison = previous.style_comparison
factual_check = previous.factual_check
length_plan = previous.length_plan
style_check = previous.style_check
guard_output = previous.guard_output
revision_messages = previous.revision_messages
_BASE_STYLE_RULES = previous.style_rules


def style_rules(mask: dict) -> str:
    base = _BASE_STYLE_RULES(mask)
    examples = mask.get("few_shot_style_examples_v26", [])
    if not examples:
        return base
    heading = (
        "\u041f\u0440\u0438\u043c\u0435\u0440\u044b \u0430\u0432\u0442\u043e\u0440\u0441\u043a\u043e\u0439 \u0444\u043e\u0440\u043c\u044b. "
        "\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439 \u0438\u0445 \u0442\u043e\u043b\u044c\u043a\u043e \u043a\u0430\u043a \u043e\u0431\u0440\u0430\u0437\u0435\u0446 \u043a\u043e\u043c\u043f\u043e\u0437\u0438\u0446\u0438\u0438, "
        "\u0440\u0438\u0442\u043c\u0430 \u0438 \u043b\u0435\u043a\u0441\u0438\u043a\u0438; \u043d\u0435 \u043f\u0435\u0440\u0435\u043d\u043e\u0441\u0438 \u0444\u0430\u043a\u0442\u044b, \u0438\u043c\u0435\u043d\u0430, \u0447\u0438\u0441\u043b\u0430 \u0438\u043b\u0438 \u0441\u043e\u0431\u044b\u0442\u0438\u044f:"
    )
    lines = [base, "", heading]
    for index, item in enumerate(examples, 1):
        tag = f"\u041f\u0420\u0418\u041c\u0415\u0420_\u0421\u0422\u0418\u041b\u042f_{index}"
        lines.append(f"<{tag}>\n{item['text']}\n</{tag}>")
    return "\n".join(lines)


def messages(row: dict, channel: str, mask: dict):
    previous_style_rules = previous.style_rules
    previous.style_rules = style_rules
    try:
        return previous.messages(row, channel, mask)
    finally:
        previous.style_rules = previous_style_rules
