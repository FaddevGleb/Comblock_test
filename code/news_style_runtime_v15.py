"""Measured composition prompts and reference-blind style scoring for v15."""

from __future__ import annotations

import copy
import statistics

import regex

import news_style_runtime_v14 as previous
import style_runtime_v3 as legacy


MODEL_ID = previous.MODEL_ID
words = previous.words
factual_check = previous.factual_check
guard_output = previous.guard_output
length_plan = previous.length_plan
load_mask = previous.load_mask


def rewrite_ratio(source: str, target: str) -> float:
    source_tokens = words(source)
    target_tokens = words(target)
    source_trigrams = {tuple(source_tokens[i:i + 3]) for i in range(max(0, len(source_tokens) - 2))}
    target_trigrams = [tuple(target_tokens[i:i + 3]) for i in range(max(0, len(target_tokens) - 2))]
    if not target_trigrams:
        return 0.0
    return 1 - sum(item in source_trigrams for item in target_trigrams) / len(target_trigrams)


def fold_mask(mask: dict, training_rows: list[dict]) -> dict:
    result = copy.deepcopy(mask)
    result["training_style_examples"] = [
        {"id": str(row.get("id", "")), "text": row["output"]}
        for row in training_rows if row.get("origin") == "supervised"
    ]
    ratios = [rewrite_ratio(row["input"], row["output"]) for row in training_rows if row.get("origin") == "supervised"]
    result["composition_v15"]["rewrite_ratio_median"] = round(statistics.median(ratios), 4) if ratios else None
    return result


def _style_examples(source: str, target: str, mask: dict) -> list[str]:
    desired = mask.get("word_count", {}).get("median", 30)
    source_set = set(words(source))
    target_norm = previous.normalized(target)
    def ranked(items: list[dict]) -> list[tuple[float, str, str]]:
        output = []
        for item in items:
            text = item.get("text", "").strip()
            if not text or previous.normalized(text) == target_norm or len(words(text)) > 75:
                continue
            word_set = set(words(text))
            overlap = len(source_set & word_set) / max(1, len(source_set | word_set))
            if overlap > 0.40:
                continue
            features = legacy.style_features(text)
            distance = abs(features["word_count"] - desired) / max(1, desired)
            distance += abs(features["paragraph_count"] - mask["paragraph_count"]["median"]) * 0.3
            output.append((distance + overlap, str(item.get("id", "")), text))
        return sorted(output)
    selected = []
    for pool in (mask.get("training_style_examples", []), mask.get("style_examples", [])):
        for _, _, text in ranked(pool):
            if text not in selected:
                selected.append(text)
                break
    return selected


def style_rules(mask: dict) -> str:
    word = mask["word_count"]
    composition = mask.get("composition_v15", {})
    paragraph = mask["paragraph_count"]["median"]
    emoji = mask["emoji"]["starts_with_emoji_rate"]
    lines = [
        f"Объём: медиана {word['median']} слов; обычный диапазон {word['p10']}–{word['p90']}.",
        f"Первый смысловой ход: короткий лид примерно {composition.get('lead_words_median', mask.get('first_phrase_words', 12))} слов. Назови событие прямо, без канцелярского вступления.",
        f"Композиция: обычно {paragraph} абзац(а/ев), {mask.get('sentence_count', {}).get('median', 2)} предложения; первая фраза несёт новость, затем детали.",
    ]
    if paragraph >= 2 or composition.get("short_lead_then_paragraph_rate", 0) >= 0.35:
        lines.append("После лида отдели пояснение пустой строкой, если оно есть.")
    if emoji >= 0.8:
        lines.append("Начни с одного эмодзи, который соответствует именно событию.")
    elif emoji <= 0.2:
        lines.append("Обычно начинай с текста, без эмодзи.")
    else:
        lines.append("Начальный эмодзи возможен лишь когда он подходит событию.")
    punct = mask.get("punctuation_per_1000_chars", {})
    lines.append(f"Наблюдаемая пунктуация на 1000 символов: тире {punct.get('dash', 0)}, кавычки {punct.get('quotation_marks', 0)}, вопросы {punct.get('question_mark', 0)}.")
    lines.append(f"Последняя фраза обычно около {composition.get('closing_words_median', 10)} слов. Не заканчивай служебным выводом или пересказом источника.")
    signature = mask.get("signature", {})
    if signature.get("rate", 0) >= 0.8 and signature.get("value"):
        lines.append(f"Подпись в конце характерна для {signature['rate']:.0%} постов: {signature['value']}.")
    return "\n".join(lines)


def messages(row: dict, channel: str, mask: dict) -> tuple[list[dict], list[dict]]:
    del channel
    source = row["input"]
    plan = length_plan(source, mask)
    examples = _style_examples(source, row.get("output", ""), mask)
    example_text = "\n\n".join(f"Образец подачи {i}:\n{text}" for i, text in enumerate(examples, 1)) or "Нет"
    system = (
        "Ты редактор Telegram-канала. Сначала установи главный подтверждённый факт, затем вырази его в наблюдаемой подаче автора. "
        "Сохраняй событие, его статус, действующих лиц, места, числа и причинность. Верни только готовый пост."
    )
    user = (
        f"<SOURCE_NEWS>\n{source}\n</SOURCE_NEWS>\n\n"
        f"<AUTHOR_STYLE_MASK>\n{style_rules(mask)}\n</AUTHOR_STYLE_MASK>\n\n"
        f"<STYLE_EXAMPLES_ONLY>\n{example_text}\n</STYLE_EXAMPLES_ONLY>\n\n"
        "<REQUIREMENTS>\n"
        "Лид сформулируй заново: начни с новости, а не с названия учреждения, аналитиков или формулы «сообщается». "
        "Выбери главное, затем добавь только необходимые детали. Не переноси факты из образцов подачи. "
        "Не придумывай имена, названия, места, даты, суммы, цитаты, решения, последствия и причины. "
        "Ироничная или образная концовка допустима, если она очевидно оценочная и не сообщает нового проверяемого факта; не добавляй её механически. "
        "Сохраняй различие между предложением, решением и уже случившимся событием. "
        f"Ориентир по длине {plan['target_words']} слов, пределы {plan['min_words']}–{plan['max_words']}. "
        "Проверь фактологию и композицию перед ответом. Выведи только пост.\n</REQUIREMENTS>"
    )
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return prompt, prompt + [{"role": "assistant", "content": row.get("output", "")}]


def _closeness(actual: float, desired: float, scale: float) -> float:
    return max(0.0, 1.0 - abs(actual - desired) / max(1e-6, scale))


def blind_style_score(source: str, candidate: str, mask: dict) -> dict:
    """Score output against independent author statistics; never read val target."""
    f = legacy.style_features(candidate)
    word = mask["word_count"]
    comp = mask.get("composition_v15", {})
    emoji_probability = float(mask["emoji"]["starts_with_emoji_rate"])
    values = {
        "length": _closeness(f["word_count"], word["median"], max(12, word["p90"] - word["p10"])),
        "lead": _closeness(f["first_phrase_words"], comp.get("lead_words_median", mask.get("first_phrase_words", 12)), 16),
        "paragraphs": _closeness(f["paragraph_count"], mask["paragraph_count"]["median"], 2),
        "sentences": _closeness(f["sentence_count"], mask.get("sentence_count", {}).get("median", 2), 4),
        "emoji": emoji_probability if f["starts_with_emoji"] else 1 - emoji_probability,
        "dash": _closeness(f["punctuation"]["—"], comp.get("dash_per_post_mean", 0), 2),
        "quotes": _closeness(f["punctuation"]["«"], comp.get("quote_per_post_mean", 0), 2),
    }
    expected_rewrite = comp.get("rewrite_ratio_median")
    if expected_rewrite is not None:
        values["rewrite"] = _closeness(rewrite_ratio(source, candidate), expected_rewrite, 0.5)
    return {"score": round(statistics.mean(values.values()), 4), "components": {key: round(value, 4) for key, value in values.items()}}


def style_check(candidate: str, mask: dict) -> dict:
    result = previous.style_check(candidate, mask)
    features = legacy.style_features(candidate)
    expected_lead = mask.get("composition_v15", {}).get("lead_words_median", mask.get("first_phrase_words", 12))
    if features["first_phrase_words"] > max(18, expected_lead * 1.6):
        result["issues"].append("lead_too_long")
    expected_paragraphs = mask.get("paragraph_count", {}).get("median", 1)
    if abs(features["paragraph_count"] - expected_paragraphs) > 1:
        result["issues"].append("paragraph_structure_mismatch")
    result["passed"] = not result["issues"]
    return result


def style_comparison(text: str, mask: dict) -> dict:
    return previous.style_comparison(text, mask)


def revision_messages(source: str, candidate: str, violations: dict, mask: dict) -> list[dict]:
    return [
        {"role": "system", "content": "Исправь пост по замечаниям. Сохраняй проверяемые факты и авторскую композицию. Выведи только пост."},
        {"role": "user", "content": f"Новость:\n{source}\n\nПост:\n{candidate}\n\nЗамечания:\n{violations}\n\nСтиль:\n{style_rules(mask)}"},
    ]
