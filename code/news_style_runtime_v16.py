"""Inference-only factuality prompt for v15 adapters.

No training logic is changed here.  The adapter and style mask are reused;
only the generation/revision instructions are stricter about source facts.
"""

from __future__ import annotations

import news_style_runtime_v15 as previous

MODEL_ID = previous.MODEL_ID
words = previous.words
factual_check = previous.factual_check
guard_output = previous.guard_output
length_plan = previous.length_plan
load_mask = previous.load_mask
fold_mask = previous.fold_mask
blind_style_score = previous.blind_style_score
style_check = previous.style_check
style_comparison = previous.style_comparison
_style_examples = previous._style_examples


def style_rules(mask: dict) -> str:
    """Keep the measured style constraints, without embedding factual content."""
    word = mask.get("word_count", {})
    comp = mask.get("composition_v15", {})
    paragraph = mask.get("paragraph_count", {}).get("median", 1)
    emoji = mask.get("emoji", {}).get("starts_with_emoji_rate", 0)
    punct = mask.get("punctuation_per_1000_chars", {})
    lines = [
        f"Длина: медиана {word.get('median', 30)} слов; обычный диапазон {word.get('p10', 8)}–{word.get('p90', 70)}.",
        f"Композиция: обычно {paragraph} абзац(а/ев), медиана {mask.get('sentence_count', {}).get('median', 2)} предложений.",
        f"Первый лид: около {comp.get('lead_words_median', mask.get('first_phrase_words', 12))} слов; сначала событие, затем только детали из исходника.",
        f"Пунктуация: тире {punct.get('dash', 0)}, кавычки {punct.get('quotation_marks', 0)}, вопросы {punct.get('question_mark', 0)} на 1000 символов.",
        f"Начальный эмодзи: наблюдаемая доля {emoji:.3f}; используй его только если он уместен и не меняет смысл.",
        f"Финальная фраза: около {comp.get('closing_words_median', 10)} слов; без нового вывода и новых фактов.",
    ]
    signature = mask.get("signature", {})
    if signature.get("rate", 0) >= 0.8 and signature.get("value"):
        lines.append(f"Подпись в конце: {signature['value']}.")
    return "\n".join(lines)


def messages(row: dict, channel: str, mask: dict) -> tuple[list[dict], list[dict]]:
    del channel
    source = row["input"]
    plan = length_plan(source, mask)
    examples = _style_examples(source, row.get("output", ""), mask)
    example_text = "\n\n".join(
        f"Пример только композиции {i}:\n{text}" for i, text in enumerate(examples, 1)
    ) or "Примеры не использовать."

    system = (
        "Ты редактор Telegram-новостей. Перепиши исходную новость, но считай исходник "
        "единственным источником фактов. Сначала мысленно выдели факты, затем напиши пост. "
        "Верни только готовый пост без объяснений, анализа и черновика."
    )
    user = (
        f"<SOURCE_NEWS>\n{source}\n</SOURCE_NEWS>\n\n"
        f"<AUTHOR_STYLE_MASK>\n{style_rules(mask)}\n</AUTHOR_STYLE_MASK>\n\n"
        f"<STYLE_EXAMPLES_ONLY>\n{example_text}\n</STYLE_EXAMPLES_ONLY>\n\n"
        "<FACTUAL_CONTRACT>\n"
        "1. Используй только сведения, которые явно есть в SOURCE_NEWS.\n"
        "2. Сохрани без изменения все числа, проценты, суммы, даты, имена, организации, места, "
        "статусы, действия и причинно-следственные связи, если они нужны для смысла.\n"
        "3. Не добавляй новых людей, организаций, мест, дат, цен, процентов, обстоятельств, "
        "причин, последствий, цитат, сравнений и юридических выводов.\n"
        "4. Не переноси факты из STYLE_EXAMPLES_ONLY в новый пост. Примеры задают только форму.\n"
        "5. Если деталь не подтверждена исходником — опусти её. Не угадывай и не заполняй пробелы.\n"
        "6. Не усиливай формулировки: «может» нельзя превращать в «произошло», а «обсуждают» — в "
        "«решили».\n"
        "7. Ирония допустима только как неинформативная стилистическая интонация и не должна "
        "создавать новый проверяемый факт.\n"
        "8. Перед выводом проверь пост по чек-листу: каждое число, имя, место и событие должно "
        "иметь опору в SOURCE_NEWS.\n"
        "</FACTUAL_CONTRACT>\n\n"
        f"<FORMAT>Ориентир: {plan['target_words']} слов; допустимо {plan['min_words']}–{plan['max_words']}. "
        "Сохрани формат автора, но точность важнее длины.</FORMAT>\n"
        "Выведи только пост."
    )
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return prompt, prompt + [{"role": "assistant", "content": row.get("output", "")}]


def revision_messages(source: str, candidate: str, violations: dict, mask: dict) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Ты выпускающий редактор. Исправь пост только по исходной новости. "
                "Удаляй неподтверждённые детали, не заменяй их новыми догадками. "
                "Верни только исправленный пост."
            ),
        },
        {
            "role": "user",
            "content": (
                f"<SOURCE_NEWS>\n{source}\n</SOURCE_NEWS>\n\n"
                f"<DRAFT>\n{candidate}\n</DRAFT>\n\n"
                f"<AUTOMATIC_REVIEW>\n{violations}\n</AUTOMATIC_REVIEW>\n\n"
                "Удалить все утверждения, которых нет в SOURCE_NEWS. Сохранить только подтверждённые "
                "факты, числа, имена и места. Не добавлять новых фактов при исправлении.\n\n"
                f"<STYLE>\n{style_rules(mask)}\n</STYLE>\n"
                "Только готовый пост, без комментариев."
            ),
        },
    ]
