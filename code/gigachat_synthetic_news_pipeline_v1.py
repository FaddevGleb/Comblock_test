"""Generate source-grounded synthetic news fragments with GigaChat.

The script keeps credentials in .env, uses GigaChat automatic built-in
functions (including web search when available), enforces JSON Schema output,
and writes both an enriched JSON document and JSONL records compatible with
the local Telegram dataset shape.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "telegram_news_dataset.jsonl"
DEFAULT_MASK_BANKI = ROOT / "style_masks" / "style_banki_oil.json"
DEFAULT_MASK_ALL = ROOT / "style_masks" / "style_masks_all.json"
DEFAULT_TRAIN_FILES = (ROOT / "train_type1.jsonl", ROOT / "train_type2.jsonl")
WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def repair_text(value: Any) -> str:
    """Repair the CP1251/UTF-8 mojibake present in older local exports."""
    text = str(value or "")
    if not text:
        return text
    try:
        repaired = text.encode("cp1251").decode("utf-8")
        if repaired.count("Р") + repaired.count("С") < text.count("Р") + text.count("С"):
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_style_masks(path_banki: Path, path_all: Path) -> dict[str, Any]:
    banki = json.loads(path_banki.read_text(encoding="utf-8"))
    all_masks = json.loads(path_all.read_text(encoding="utf-8"))
    return {"banki_oil": banki, "all": all_masks}


def channel_mask(masks: dict[str, Any], channel: str) -> dict[str, Any]:
    if channel == "banki_oil":
        return masks["banki_oil"]
    channels = masks["all"].get("channels", {})
    if channel in channels:
        return channels[channel]
    return masks["all"]


def compact_style_mask(mask: dict[str, Any]) -> dict[str, Any]:
    """Keep style evidence compact enough for every generation request."""
    wanted = {
        "corpus": ["document_count", "language"],
        "length": ["words_per_document", "sentences_per_document", "paragraphs_per_document"],
        "sentences": ["words_per_sentence", "sentence_types", "first_phrase"],
        "paragraphs": ["single_sentence_paragraph_rate", "short_lead_then_explanation_rate"],
        "punctuation": ["per_1000_chars", "repeated_punctuation_rate"],
        "capitalization": ["uppercase_letter_rate", "all_caps_word_rate"],
        "formatting": ["list_rate", "url_rate", "mention_rate", "emoji_rate_per_1000_words", "top_emojis"],
        "characteristic_patterns": ["repeated_phrases", "sentence_openers", "sentence_closers"],
    }
    result: dict[str, Any] = {}
    for section, keys in wanted.items():
        source = mask.get(section, {})
        if isinstance(source, dict):
            result[section] = {key: source[key] for key in keys if key in source}
    return result


def few_shots(train_rows: list[dict[str, Any]], channel: str, limit: int = 2) -> list[dict[str, str]]:
    selected = []
    for row in train_rows:
        instruction = repair_text(row.get("instruction", ""))
        if channel not in instruction and channel not in repair_text(row.get("output", "")):
            continue
        source = repair_text(row.get("input", "")).strip()
        target = repair_text(row.get("output", "")).strip()
        if source and target:
            selected.append({"source": source, "target": target})
        if len(selected) >= limit:
            break
    return selected


def style_features(text: str) -> dict[str, Any]:
    words = WORD_RE.findall(text)
    return {
        "word_count": len(words),
        "char_count": len(text),
        "sentence_count": len(re.findall(r"[.!?…]+", text)),
        "paragraph_count": len([part for part in re.split(r"\n\s*\n", text) if part.strip()]),
        "line_break_count": text.count("\n"),
        "emoji_count": len(re.findall(r"[^\x00-\x7F\w\s]", text, re.UNICODE)),
        "has_mention": bool(re.search(r"@[\w_]+", text)),
    }


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "fragment": {"type": "string", "description": "Готовый русский Telegram-пост, только текст новости."},
            "fact_basis": {"type": "array", "items": {"type": "string"}, "description": "Короткие атомарные факты из исходного поста или проверенного поиска."},
            "search_used": {"type": "boolean"},
            "search_summary": {"type": "string", "description": "Кратко: что подтверждено поиском; пустая строка, если поиск недоступен."},
        },
        "required": ["fragment", "fact_basis", "search_used", "search_summary"],
    }


class GigaChatClient:
    def __init__(self) -> None:
        self.auth_key = os.environ["GIGACHAT_AUTH_KEY"]
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.model = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")
        self.base_url = os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat").rstrip("/")
        self.oauth_url = os.getenv("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
        self.verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}
        self.timeout = int(os.getenv("GIGACHAT_TIMEOUT", "180"))
        self.session = requests.Session()
        self.access_token = ""
        self.expires_at = 0.0

    def token(self) -> str:
        if self.access_token and time.time() < self.expires_at - 60:
            return self.access_token
        response = self.session.post(
            self.oauth_url,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {self.auth_key}",
            },
            data={"scope": self.scope},
            timeout=60,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        data = response.json()
        self.access_token = data["access_token"]
        self.expires_at = float(data.get("expires_at", time.time() + 1800))
        return self.access_token

    def complete(self, messages: list[dict[str, str]], retries: int = 4) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(os.getenv("GIGACHAT_TEMPERATURE", "0.2")),
            "top_p": 0.9,
            "function_call": "auto",
            "functions": [],
            "response_format": {"type": "json_schema", "schema": schema(), "strict": True},
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json", "Accept": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                if response.status_code == 401:
                    self.access_token = ""
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                parsed = json.loads(content)
                if not isinstance(parsed, dict) or not parsed.get("fragment"):
                    raise ValueError("GigaChat returned an invalid structured response")
                return {"response": parsed, "raw": data}
            except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(f"GigaChat request failed after {retries} attempts: {last_error}") from last_error


def system_prompt(channel: str, mask: dict[str, Any], shots: list[dict[str, str]]) -> str:
    examples = "\n\n".join(
        f"Пример {index}:\nИСХОДНЫЕ ФАКТЫ:\n{item['source']}\nГОТОВЫЙ ПОСТ:\n{item['target']}"
        for index, item in enumerate(shots, 1)
    )
    return f"""Ты — редактор Telegram-канала {channel}. Создавай синтетические обрывки новостей только на основе проверяемых фактов.

ОБЯЗАТЕЛЬНО:
1. Используй встроенный веб-поиск GigaChat в автоматическом режиме, если исходный пост содержит проверяемое событие, имя, дату, число или организацию.
2. Не добавляй ни одного факта, числа, имени, места, причины или последствия, которых нет в исходном посте либо в подтверждённом результате поиска.
3. Если факт не подтверждён, не конкретизируй его и не выдумывай замену.
4. Сохраняй смысл и все исходные числа. Не меняй знак числа, дату и единицы измерения.
5. Пиши только готовый пост на русском языке. Без пояснений, заголовка «Ответ», ссылок и служебных комментариев.
6. Формат поста, длина, пунктуация, эмодзи и композиция должны соответствовать маске автора, но стиль не должен менять факты.
7. Не копируй дословно исходный текст: сделай аккуратную редакторскую вариацию.

Маска стиля канала {channel}:
{json.dumps(compact_style_mask(mask), ensure_ascii=False, indent=2)}

Few-shot примеры:
{examples or 'Нет доступных примеров; следуй маске и ограничениям выше.'}

Верни только JSON по заданной схеме. Поле fact_basis должно перечислять только факты, реально использованные в fragment."""


def make_record(source_row: dict[str, Any], generated: dict[str, Any], mask: dict[str, Any], raw: dict[str, Any], model: str) -> dict[str, Any]:
    channel = source_row["channel"]
    source_text = repair_text(source_row.get("telegram", {}).get("text", ""))
    fragment = generated["fragment"].strip()
    return {
        "schema_version": "1.0",
        "id": f"synthetic:{channel}:{source_row.get('id', uuid.uuid4().hex)}",
        "label": source_row.get("label", "synthetic_news"),
        "channel": channel,
        "telegram": {"message_id": None, "published_at": datetime.now(timezone.utc).isoformat(), "text": fragment, "views": None, "forwards": None, "reply_to_message_id": None},
        "style_features": style_features(fragment),
        "source_real": {"id": source_row.get("id"), "text": source_text},
        "style_mask": mask,
        "search": {"used": bool(generated.get("search_used")), "summary": generated.get("search_summary", ""), "fact_basis": generated.get("fact_basis", [])},
        "synthetic": {"provider": "GigaChat", "model": model, "prompt_version": "gigachat_synthetic_news_v1", "raw_finish_reason": raw.get("choices", [{}])[0].get("finish_reason")},
    }


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--channel", choices=["banki_oil", "moscowach"])
    parser.add_argument("--output", type=Path, default=ROOT / "synthetic_news_gigachat_v1.json")
    parser.add_argument("--jsonl-output", type=Path, default=ROOT / "synthetic_news_gigachat_v1.jsonl")
    args = parser.parse_args()

    required = ("GIGACHAT_AUTH_KEY",)
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise SystemExit(f"Missing .env variables: {', '.join(missing)}")

    rows = read_jsonl(args.dataset)
    rows = [row for row in rows if not args.channel or row.get("channel") == args.channel][:args.limit]
    masks = load_style_masks(DEFAULT_MASK_BANKI, DEFAULT_MASK_ALL)
    train_rows = [row for path in DEFAULT_TRAIN_FILES if path.exists() for row in read_jsonl(path)]
    client = GigaChatClient()
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        channel = row["channel"]
        mask = channel_mask(masks, channel)
        source_text = repair_text(row.get("telegram", {}).get("text", ""))
        messages = [
            {"role": "system", "content": system_prompt(channel, mask, few_shots(train_rows, channel))},
            {"role": "user", "content": f"ИСХОДНЫЙ ПОСТ:\n{source_text}\n\nСделай одну синтетическую редакторскую вариацию. Сохрани только подтверждаемые факты."},
        ]
        result = client.complete(messages)
        output_rows.append(make_record(row, result["response"], mask, result["raw"], client.model))
        print(f"[{index}/{len(rows)}] {channel}: generated", flush=True)

    document = {
        "schema_version": "synthetic_news_gigachat_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.dataset),
        "model": client.model,
        "style_masks": masks,
        "records": output_rows,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    args.jsonl_output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n", encoding="utf-8")
    # Build type-specific mixed corpora: original records first, synthetic
    # records second. The original record shape is preserved for both parts.
    real_by_type: dict[str, list[dict[str, Any]]] = {"type1": [], "type2": []}
    for row in rows:
        label = row.get("label")
        if label in real_by_type:
            real_by_type[label].append(row)
    synthetic_by_type: dict[str, list[dict[str, Any]]] = {"type1": [], "type2": []}
    for row in output_rows:
        label = row.get("label")
        if label in synthetic_by_type:
            synthetic_by_type[label].append(row)
    for label in ("type1", "type2"):
        mixed = {
            "schema_version": "synth_and_real_training_data_v1",
            "label": label,
            "created_at": document["created_at"],
            "source_dataset": str(args.dataset),
            "model": client.model,
            "style_masks": masks,
            "real_count": len(real_by_type[label]),
            "synthetic_count": len(synthetic_by_type[label]),
            "records": real_by_type[label] + synthetic_by_type[label],
        }
        type_path = ROOT / f"synth_and_real_training_data_{label}.json"
        type_path.write_text(json.dumps(mixed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(output_rows)} records to {args.output} and {args.jsonl_output}")


if __name__ == "__main__":
    main()
