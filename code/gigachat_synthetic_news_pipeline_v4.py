"""Build training pairs by reconstructing a neutral source news from a Telegram post.

The target is always the original channel post. GigaChat generates only the
input/source news, without adding facts that are absent from the post.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import gigachat_synthetic_news_pipeline_v1 as v1


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "gigachat_reverse_news_v4_progress.jsonl"


def parse_response(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    text = str(content or "").strip()
    candidates = [text]
    if "```" in text:
        candidates.append(text.replace("```json", "").replace("```", "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            for key in ("source_news", "fragment", "news"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return data[key].strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    raise ValueError(f"GigaChat returned invalid source news: {text[:300]!r}")


def request(client: v1.GigaChatClient, messages: list[dict[str, str]], retries: int = 5) -> str:
    payload = {
        "model": client.model,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.8,
        "response_format": {"type": "json_object"},
    }
    error = None
    for attempt in range(retries):
        try:
            response = client.session.post(
                f"{client.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {client.token()}", "Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=client.timeout,
                verify=client.verify_ssl,
            )
            response.raise_for_status()
            message = response.json().get("choices", [{}])[0].get("message", {})
            return parse_response(message.get("content"))
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"request failed after {retries} attempts: {error}") from error


def instruction_for(rows: list[dict], channel: str) -> str:
    candidates = [str(row.get("instruction", "")).strip() for row in rows if channel in str(row.get("instruction", ""))]
    return candidates[0] if candidates else f"Перепиши новостной текст в стиле Telegram-канала @{channel}."


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "telegram_news_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if not os.getenv("GIGACHAT_AUTH_KEY"):
        raise SystemExit("Set GIGACHAT_AUTH_KEY in .env or environment")

    source_rows = v1.read_jsonl(args.dataset)[: args.limit]
    train_rows = [row for path in (ROOT / "train_type1.jsonl", ROOT / "train_type2.jsonl") if path.exists() for row in v1.read_jsonl(path)]
    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    completed = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["source_id"]] = item

    client = v1.GigaChatClient()
    with CHECKPOINT.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(source_rows, 1):
            source_id = str(row.get("id", index))
            if source_id in completed:
                continue
            channel = str(row.get("channel", ""))
            label = "type1" if channel == "banki_oil" else "type2"
            post = v1.repair_text(row.get("telegram", {}).get("text", "")).strip()
            instruction = instruction_for(train_rows, channel)
            examples = [r for r in train_rows if channel in str(r.get("instruction", ""))][:2]
            few_shot = "\n\n".join(f"ПОСТ:\n{r['output']}\nИСХОДНАЯ НОВОСТЬ:\n{r['input']}" for r in examples)
            messages = [
                {"role": "system", "content": 'Ты восстанавливаешь исходную новость по опубликованному Telegram-посту. Верни только JSON {"source_news":"..."}. Исходная новость должна быть нейтральной, без стилистики канала. Используй только факты, числа, имена, места и действия, явно присутствующие в посте. Не добавляй причины, источники, оценки, даты и детали от себя. Если пост содержит шутку или оценку, сохрани только проверяемую фактическую основу.'},
                {"role": "user", "content": f"КАНАЛ: @{channel}\nИНСТРУКЦИЯ ДЛЯ ОБУЧЕНИЯ: {instruction}\n\nFEW-SHOT:\n{few_shot}\n\nПОСТ ИЗ TELEGRAM:\n{post}\n\nСформируй одну исходную новость для поля input. Верни строгий JSON с ключом source_news."},
            ]
            try:
                news = request(client, messages)
                item = {"source_id": source_id, "ok": True, "record": {"input": news, "output": post, "label": label}}
                print(f"[{index}/{len(source_rows)}] {channel}: generated", flush=True)
            except Exception as exc:
                item = {"source_id": source_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{index}/{len(source_rows)}] {channel}: failed: {exc}", flush=True)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            completed[source_id] = item

    records = [item["record"] for item in completed.values() if item.get("ok")]
    for label in ("type1", "type2"):
        rows = [r for r in records if r["label"] == label]
        clean_rows = [{"input": r["input"], "output": r["output"]} for r in rows]
        (ROOT / f"synth_and_real_training_data_{label}_v2.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean_rows) + "\n", encoding="utf-8")
    clean_records = [{"input": r["input"], "output": r["output"]} for r in records]
    (ROOT / "synthetic_news_reverse_v4.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in clean_records) + "\n", encoding="utf-8")
    print(f"Finished: records={len(records)} errors={len(source_rows)-len(records)}")


if __name__ == "__main__":
    main()
