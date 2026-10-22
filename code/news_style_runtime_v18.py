"""One prompt contract for QLoRA training and inference in v18."""

from __future__ import annotations

import re
import regex

import news_style_runtime_v15 as prior

MODEL_ID = prior.MODEL_ID
words = prior.words
load_mask = prior.load_mask
blind_style_score = prior.blind_style_score
style_comparison = prior.style_comparison


def fold_mask(mask: dict, training_rows: list[dict]) -> dict:
    # Style statistics come from the independent channel corpus.
    del training_rows
    return mask

NUMBER = regex.compile(r"(?<![\p{L}\p{N}])\d+(?:[\s\u00a0]\d{3})*(?:[,.]\d+)?\s*%?")


def numbers(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group()).replace(",", ".") for match in NUMBER.finditer(text)}


def factual_check(source: str, candidate: str) -> dict:
    unexpected = sorted(numbers(candidate) - numbers(source))
    return {
        "unsupported_numbers": unexpected,
        "deterministic_passed": not unexpected,
        "needs_review": bool(unexpected),
    }


def length_plan(source: str, mask: dict) -> dict:
    stats = mask["length_v18"]
    mean, sigma = stats["mean"], stats["std"]
    lower = stats["min_words"]
    upper = stats["max_words"]
    # The full author range is mean +/- one sigma. Aim near the centre to
    # counter the previous tendency to collapse posts into a single sentence.
    target = max(lower, min(upper, round(mean)))
    return {
        "source_words": len(words(source)),
        "target_words": target,
        "min_words": lower,
        "max_words": upper,
        "mean_words": mean,
        "std_words": sigma,
        "preferred_min_words": max(lower, round(mean * 0.7)),
    }


def style_rules(mask: dict) -> str:
    length = mask["length_v18"]
    style = mask["style_v18"]
    emoji = mask.get("emoji", {})
    composition = mask.get("composition_v15", {})
    signature = mask.get("signature", {})
    lines = [
        f"Длина авторских постов: среднее {length['mean']:.1f} слов, σ={length['std']:.1f}; обычный коридор μ±σ: {length['min_words']}–{length['max_words']} слов.",
        f"Абзацы: в среднем {style['paragraphs_mean']:.1f}; предложения: в среднем {style['sentence_mean']:.1f}.",
        f"Доля постов с эмодзи в начале: {style['initial_emoji_rate']:.1%}; добавляй один уместный эмодзи только если он помогает передать тему.",
        f"Средняя длина лида около {composition.get('lead_words_median', mask.get('first_phrase_words', 12))} слов.",
        f"Короткий лид с пояснением через абзац: {composition.get('short_lead_then_paragraph_rate', 0):.1%} постов.",
        f"Средняя длина заключительной фразы: {composition.get('closing_words_median', 0):.1f} слов; начало с кавычек: {composition.get('quote_start_rate', 0):.1%}.",
        f"Тире: {composition.get('dash_per_post_mean', 0):.2f} на пост; кавычки: {composition.get('quote_per_post_mean', 0):.2f} на пост.",
        "Передавай узнаваемую композицию и ритм автора; содержание и оценка должны следовать только из исходной новости.",
    ]
    if signature.get("rate", 0) >= 0.8 and signature.get("value"):
        lines.append(f"Авторская подпись: {signature['value']}.")
    return "\n".join(lines)


def messages(row: dict, channel: str, mask: dict) -> tuple[list[dict], list[dict]]:
    source = row["input"]
    length = length_plan(source, mask)
    system = (
        f"Ты редактор Telegram-канала {channel}. Перепиши новость в авторской форме. "
        "Единственный источник сведений — блок ИСХОДНАЯ_НОВОСТЬ. "
        "Факты важнее длины и выразительности. Верни только готовый пост."
    )
    user = (
        f"<ИСХОДНАЯ_НОВОСТЬ>\n{source}\n</ИСХОДНАЯ_НОВОСТЬ>\n\n"
        f"<СТИЛЬ_АВТОРА>\n{style_rules(mask)}\n</СТИЛЬ_АВТОРА>\n\n"
        "<ТРЕБОВАНИЯ_К_СМЫСЛУ>\n"
        "Сначала мысленно выдели опорные факты: кто, что сделал, где, когда, сколько, статус события и причины. "
        "Переноси только то, что явно следует из исходной новости. "
        "Числа, проценты, суммы, даты и имена передавай точно, без округления и подмены. "
        "Не добавляй новые места, участников, действия, причины, последствия, решения, сравнения, цитаты или атрибуцию. "
        "Запрещены правдоподобные догадки и факты из других новостей. "
        "Не превращай предположение в случившееся событие. "
        "Ирония допустима только если она не утверждает новый проверяемый факт. "
        "Если в исходнике нет материала для длинного поста, оставь пост короче; не заполняй длину выдумками. "
        "Проверь каждое утверждение перед ответом.\n"
        "</ТРЕБОВАНИЯ_К_СМЫСЛУ>\n\n"
        f"<ДЛИНА>Ориентир {length['target_words']} слов; типичный коридор {length['min_words']}–{length['max_words']}; "
        f"при достаточном количестве фактов старайся написать не менее {length['preferred_min_words']} слов.</ДЛИНА>\n\n"
        "Напиши только новый пост."
    )
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return prompt, prompt + [{"role": "assistant", "content": row.get("output", "")}]


def style_check(candidate: str, mask: dict) -> dict:
    from style_runtime_v3 import style_features
    features = style_features(candidate)
    length = mask["length_v18"]
    issues = []
    if not candidate.strip():
        issues.append("empty")
    if "<think>" in candidate or "</think>" in candidate:
        issues.append("service_tokens")
    if features["word_count"] < length["min_words"]:
        issues.append("too_short")
    if features["word_count"] > length["max_words"]:
        issues.append("too_long")
    signature = mask.get("signature", {})
    if signature.get("rate", 0) >= 0.8 and signature.get("value"):
        if not candidate.rstrip().endswith(signature["value"]):
            issues.append("signature_mismatch")
    return {"passed": not issues, "issues": issues}


def guard_output(source: str, output: str, channel: str, mask: dict) -> tuple[str, dict]:
    del channel, mask
    return output, factual_check(source, output)


def revision_messages(source: str, candidate: str, violations: dict, mask: dict) -> list[dict]:
    return [
        {"role": "system", "content": "Исправь черновик по исходной новости. Удали каждую неподтверждённую деталь. Не заменяй её новой догадкой. Верни только пост."},
        {"role": "user", "content": (
            f"<ИСХОДНАЯ_НОВОСТЬ>\n{source}\n</ИСХОДНАЯ_НОВОСТЬ>\n\n"
            f"<ЧЕРНОВИК>\n{candidate}\n</ЧЕРНОВИК>\n\n"
            f"<ПРОБЛЕМЫ>\n{violations}\n</ПРОБЛЕМЫ>\n\n"
            f"<СТИЛЬ>\n{style_rules(mask)}\n</СТИЛЬ>\n"
            "Проверь точность чисел, имён, мест, статуса, причины и последствий. "
            "Удали неподтверждённые фразы. Сохрани имеющиеся факты и авторскую композицию."
        )},
    ]
