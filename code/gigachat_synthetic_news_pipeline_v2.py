"""Resumable, fault-tolerant v2 runner for the GigaChat synthetic corpus."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import gigachat_synthetic_news_pipeline_v1 as v1


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "synthetic_news_gigachat_v2_progress.jsonl"


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = str(content or "").strip()
    if not text:
        raise ValueError("empty assistant content")
    candidates = [text]
    if "```" in text:
        candidates.append(text.replace("```json", "").replace("```", "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and isinstance(parsed.get("fragment"), str) and parsed["fragment"].strip():
                parsed.setdefault("fact_basis", [])
                parsed.setdefault("search_used", False)
                parsed.setdefault("search_summary", "")
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"invalid JSON content: {text[:300]!r}")


def complete(client: v1.GigaChatClient, messages: list[dict[str, str]], retries: int = 5) -> dict[str, Any]:
    payload = {
        "model": client.model,
        "messages": messages,
        "temperature": float(os.getenv("GIGACHAT_TEMPERATURE", "0.2")),
        "top_p": 0.9,
        "function_call": "auto",
        "functions": [],
        "response_format": {"type": "json_schema", "schema": v1.schema(), "strict": True},
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.session.post(
                f"{client.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {client.token()}", "Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
                timeout=client.timeout,
                verify=client.verify_ssl,
            )
            if response.status_code == 401:
                client.access_token = ""
            response.raise_for_status()
            raw = response.json()
            message = raw.get("choices", [{}])[0].get("message", {})
            return {"response": parse_json_content(message.get("content")), "raw": raw}
        except Exception as exc:  # one malformed response must not kill the corpus job
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}") from last_error


def write_type_files(real_rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]], masks: dict[str, Any], model: str, created_at: str) -> None:
    for label in ("type1", "type2"):
        real = [row for row in real_rows if row.get("label") == label]
        synthetic = [row for row in synthetic_rows if row.get("label") == label]
        document = {
            "schema_version": "synth_and_real_training_data_v2",
            "label": label,
            "created_at": created_at,
            "model": model,
            "real_count": len(real),
            "synthetic_count": len(synthetic),
            "style_masks": masks,
            "records": real + synthetic,
        }
        (ROOT / f"synth_and_real_training_data_{label}.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=v1.DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--channel", choices=["banki_oil", "moscowach"])
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if not os.getenv("GIGACHAT_AUTH_KEY"):
        raise SystemExit("Set GIGACHAT_AUTH_KEY in .env or the environment")

    source_rows = v1.read_jsonl(args.dataset)
    source_rows = [row for row in source_rows if not args.channel or row.get("channel") == args.channel][:args.limit]
    masks = v1.load_style_masks(v1.DEFAULT_MASK_BANKI, v1.DEFAULT_MASK_ALL)
    train_rows = [row for path in v1.DEFAULT_TRAIN_FILES if path.exists() for row in v1.read_jsonl(path)]
    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    completed: dict[str, dict[str, Any]] = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["source_id"]] = item

    client = v1.GigaChatClient()
    created_at = datetime.now(timezone.utc).isoformat()
    with CHECKPOINT.open("a", encoding="utf-8") as progress:
        for index, source_row in enumerate(source_rows, 1):
            source_id = str(source_row.get("id", index))
            if source_id in completed:
                print(f"[{index}/{len(source_rows)}] {source_row.get('channel')}: skipped checkpoint", flush=True)
                continue
            channel = source_row["channel"]
            mask = v1.channel_mask(masks, channel)
            source_text = v1.repair_text(source_row.get("telegram", {}).get("text", ""))
            messages = [
                {"role": "system", "content": v1.system_prompt(channel, mask, v1.few_shots(train_rows, channel))},
                {"role": "user", "content": f"ИСХОДНЫЙ ПОСТ:\n{source_text}\n\nСделай одну синтетическую редакторскую вариацию. Сохрани только подтверждаемые факты."},
            ]
            try:
                result = complete(client, messages)
                record = v1.make_record(source_row, result["response"], mask, result["raw"], client.model)
                item = {"source_id": source_id, "ok": True, "record": record}
                print(f"[{index}/{len(source_rows)}] {channel}: generated", flush=True)
            except Exception as exc:
                item = {"source_id": source_id, "ok": False, "channel": channel, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{index}/{len(source_rows)}] {channel}: failed, continuing: {exc}", flush=True)
            progress.write(json.dumps(item, ensure_ascii=False) + "\n")
            progress.flush()
            completed[source_id] = item

    synthetic = [item["record"] for item in completed.values() if item.get("ok")]
    errors = [item for item in completed.values() if not item.get("ok")]
    real_rows = source_rows
    final = {
        "schema_version": "synthetic_news_gigachat_v2",
        "created_at": created_at,
        "source_dataset": str(args.dataset),
        "model": client.model,
        "style_masks": masks,
        "real_count": len(real_rows),
        "synthetic_count": len(synthetic),
        "error_count": len(errors),
        "errors": errors,
        "records": synthetic,
    }
    (ROOT / "synthetic_news_gigachat_v2.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "synthetic_news_gigachat_v2.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in synthetic) + "\n", encoding="utf-8")
    write_type_files(real_rows, synthetic, masks, client.model, created_at)
    print(f"Finished: synthetic={len(synthetic)}, errors={len(errors)}")


if __name__ == "__main__":
    main()
