"""Channel-independent prompts, style retrieval and conservative factual checks."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal

import regex

import style_runtime_v3 as legacy


MODEL_ID = legacy.MODEL_ID
WORD_RE = regex.compile(r"[\p{L}\p{N}]+(?:[-'][\p{L}\p{N}]+)?")
NUM_RE = regex.compile(r"(?<![\p{L}\p{N}])\d[\d\s\u00a0]*(?:[,.]\d+)?(?:\s*[-–—]\s*\d[\d\s\u00a0]*(?:[,.]\d+)?)?(?:\s*(?:тыс\.?|тысяч[аи]?|млн|миллион[а-я]*|млрд))?(?:\s*%)?(?![\p{L}\p{N}])", regex.I)
GRAPHEMES = regex.compile(r"\X")
EMOJI = regex.compile(r"[\p{Extended_Pictographic}\p{Regional_Indicator}]")
RUSSIAN_NUMBER_WORDS = {
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "двух": 2, "два": 2, "две": 2,
    "трех": 3, "трёх": 3, "три": 3, "четыре": 4, "четырех": 4, "четырёх": 4,
    "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10,
    "десяти": 10, "двадцать": 20, "двадцати": 20, "тридцать": 30,
    "тридцати": 30, "сорок": 40, "пятьдесят": 50, "сто": 100,
}


def words(text: str) -> list[str]:
    return [word.casefold() for word in WORD_RE.findall(text)]


def normalized(text: str) -> str:
    return " ".join(words(text))


def numeric_values(text: str) -> list[dict]:
    result = []
    for match in NUM_RE.finditer(text):
        raw = match.group().strip()
        body = re.sub(r"\s*%\s*$", "", raw)
        multiplier = Decimal(1)
        if re.search(r"\b(?:тыс\.?|тысяч[аи]?)\b", body, re.I):
            multiplier = Decimal(1000)
        elif re.search(r"\b(?:млн|миллион[а-я]*)\b", body, re.I):
            multiplier = Decimal(1000000)
        elif re.search(r"\bмлрд\b", body, re.I):
            multiplier = Decimal(1000000000)
        number_part = re.match(r"[\d\s\u00a0,.–—-]+", body)
        if not number_part:
            continue
        part = number_part.group().strip()
        boundaries = re.split(r"\s*[-–—]\s*", part)
        values = []
        for boundary in boundaries:
            compact = re.sub(r"[\s\u00a0]", "", boundary).replace(",", ".")
            try:
                values.append(Decimal(compact) * multiplier)
            except ArithmeticError:
                pass
        if values:
            result.append({"raw": raw, "values": values, "percent": "%" in raw, "span": match.span()})
    return result


def unsupported_numbers(source: str, candidate: str) -> list[str]:
    source_items = numeric_values(source)
    source_words = {Decimal(RUSSIAN_NUMBER_WORDS[word]) for word in words(source) if word in RUSSIAN_NUMBER_WORDS}
    unsupported = []
    for item in numeric_values(candidate):
        for value in item["values"]:
            found = not item["percent"] and value in source_words
            for original in source_items:
                if item["percent"] != original["percent"]:
                    continue
                for source_value in original["values"]:
                    if value == source_value or (value == value.to_integral_value() and abs(value - source_value) <= Decimal("0.5")):
                        found = True
                        break
                if found:
                    break
            if not found:
                unsupported.append(item["raw"])
                break
    return sorted(set(unsupported))


def unsupported_quoted_names(source: str, candidate: str) -> list[str]:
    source_norm = normalized(source)
    source_stems = {word[:5] for word in words(source) if len(word) >= 5}
    names = []
    for name in re.findall(r"[«\"]([^«»\"]{3,80})[»\"]", candidate):
        name_words = words(name)
        supported_by_tokens = bool(name_words) and all(
            word in words(source) if len(word) < 5 else word[:5] in source_stems
            for word in name_words
        )
        if normalized(name) not in source_norm and not supported_by_tokens:
            names.append(name)
    return sorted(set(names))


def factual_check(source: str, candidate: str) -> dict:
    numbers = unsupported_numbers(source, candidate)
    quoted = unsupported_quoted_names(source, candidate)
    return {
        "unsupported_numbers": numbers,
        "unsupported_quoted_names": quoted,
        "deterministic_passed": not numbers and not quoted,
        "semantic_status": "not_checked",
        "needs_review": bool(numbers or quoted),
    }


def style_check(candidate: str, mask: dict) -> dict:
    features = legacy.style_features(candidate)
    issues = []
    if not candidate.strip():
        issues.append("empty_post")
    if "<think>" in candidate or "</think>" in candidate:
        issues.append("service_tokens")
    emoji_rate = float(mask.get("emoji", {}).get("starts_with_emoji_rate") or 0)
    if emoji_rate >= 0.8 and not features["starts_with_emoji"]:
        issues.append("missing_initial_emoji")
    signature = mask.get("signature", {})
    expected = signature.get("value")
    if expected and signature.get("rate", 0) >= 0.8:
        actual = re.findall(r"@[A-Za-z_][A-Za-z0-9_]*", candidate)
        if actual and any(value != expected for value in actual):
            issues.append("wrong_signature")
        elif not actual:
            issues.append("missing_signature")
    count = mask.get("word_count", {})
    if count and not count.get("p10", 0) <= features["word_count"] <= count.get("p90", 100000):
        issues.append("length_outside_style_range")
    return {"passed": not issues, "issues": issues}


def guard_output(source: str, output: str, channel: str, mask: dict) -> tuple[str, dict]:
    del channel, mask
    check = factual_check(source, output)
    return output, {"fallback_used": False, **check}


def style_examples(source: str, mask: dict, count: int = 2) -> list[str]:
    query = set(words(source))
    candidates = []
    for item in mask.get("style_examples", []):
        text = item.get("text", "")
        tokens = set(words(text))
        if not text or not tokens or len(words(text)) > 50:
            continue
        overlap = len(query & tokens) / max(1, len(query | tokens))
        candidates.append((overlap, item.get("id", ""), text))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, _, text in candidates[:count]]


def style_rules(mask: dict) -> str:
    count = mask.get("word_count", {})
    paragraphs = mask.get("paragraph_count", {})
    emoji = mask.get("emoji", {})
    lines = [
        f"Слов на пост: обычно около {count.get('median', 30)}, диапазон P10–P90 {count.get('p10', 10)}–{count.get('p90', 60)}.",
        f"Абзацев обычно {paragraphs.get('median', 1)}; первая фраза около {mask.get('first_phrase_words', 12)} слов.",
    ]
    rate = float(emoji.get("starts_with_emoji_rate") or 0)
    if rate >= 0.8:
        lines.append("Обычно пост начинается с одного уместного по смыслу эмодзи.")
    elif rate <= 0.2:
        lines.append("Обычно пост начинается с текста без эмодзи.")
    else:
        lines.append("Эмодзи в начале используется иногда и только по смыслу.")
    examples = emoji.get("examples") or []
    if examples and rate > 0.2:
        lines.append("В корпусе встречаются эмодзи: " + " ".join(examples[:8]) + ".")
    first = mask.get("first_paragraph_words", {})
    if first:
        lines.append(f"Первый абзац: обычно около {first.get('median')} слов.")
    signature = mask.get("signature", {})
    if signature.get("rate", 0) >= 0.8 and signature.get("value"):
        lines.append(f"В корпусе подпись в конце встречается в {signature['rate']:.0%} постов: {signature['value']}.")
    return "\n".join(lines)


def length_plan(source: str, mask: dict) -> dict:
    source_count = len(words(source))
    stats = mask.get("word_count", {})
    median = int(stats.get("median", 30))
    floor = int(stats.get("p10", 10))
    ceiling = int(stats.get("p90", 60))
    # Preserve more detail in fact-dense input while staying within the author's range.
    target = max(floor, min(ceiling, max(median, int(source_count * 0.42))))
    return {"source_words": source_count, "target_words": target, "min_words": floor, "max_words": ceiling}


def source_facts(source: str) -> str:
    sentences = [s.strip() for s in regex.split(r"(?<=[.!?])\s+|\n+", source) if s.strip()]
    numbers = [item["raw"] for item in numeric_values(source)]
    return "Первая фраза источника: " + (sentences[0] if sentences else source) + "\nЧисла и даты из источника: " + (", ".join(numbers) or "нет")


def messages(row: dict, channel: str, mask: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    del channel  # Adapter selection is performed by the caller.
    source = row["input"]
    plan = length_plan(source, mask)
    examples = style_examples(source, mask)
    examples_text = "\n\n".join(f"Пример {i}:\n{text}" for i, text in enumerate(examples, 1)) or "Нет"
    system = "Ты редактор Telegram-постов. Перепиши новость в переданном стиле, сохраняя смысл и проверяемые факты. Верни только пост."
    user = (
        f"<SOURCE_NEWS>\n{source}\n</SOURCE_NEWS>\n\n"
        f"<SOURCE_FACTS>\n{source_facts(source)}\n</SOURCE_FACTS>\n\n"
        f"<AUTHOR_STYLE_MASK>\n{style_rules(mask)}\n</AUTHOR_STYLE_MASK>\n\n"
        f"<STYLE_EXAMPLES_ONLY>\n{examples_text}\n</STYLE_EXAMPLES_ONLY>\n\n"
        "<SEMANTIC_REQUIREMENTS>\n"
        "Напиши один готовый пост. Начни с главного события, затем добавь только важные детали из SOURCE_NEWS. "
        "Не добавляй новые имена, организации, суммы, даты, события, причины, последствия и цитаты. "
        "Различай предложение, принятое решение и совершившееся событие. "
        "Примеры показывают только подачу; факты из них использовать запрещено. "
        "Ирония допустима только как оценка или очевидная метафора без нового проверяемого утверждения. "
        f"Ориентир по длине: {plan['target_words']} слов, допустимо {plan['min_words']}–{plan['max_words']}. "
        "Если деталь не подтверждена источником, опусти её. Не пиши анализ или служебный текст.\n"
        "</SEMANTIC_REQUIREMENTS>\n\n<NEW_POST>\n"
    )
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return prompt, prompt + [{"role": "assistant", "content": row.get("output", "")}]


def revision_messages(source: str, candidate: str, violations: dict, mask: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Исправь только неподтверждённые сведения в готовом посте. Сохрани стиль, не добавляй новых фактов. Верни только исправленный пост."},
        {"role": "user", "content": f"Исходная новость:\n{source}\n\nПост:\n{candidate}\n\nОбнаруженные нарушения:\n{json.dumps(violations, ensure_ascii=False)}\n\nСтилевые ориентиры:\n{style_rules(mask)}"},
    ]


def style_comparison(text: str, mask: dict) -> dict:
    return legacy.style_comparison(text, mask)


def load_mask(path) -> dict:
    return legacy.load_mask(path)
