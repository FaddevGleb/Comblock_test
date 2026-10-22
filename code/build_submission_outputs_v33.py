"""Create the required plain-text outputs from the saved five-input inference run."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "kaggle_download_inference_v30" / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
for channel, filename in (("banki_oil", "outputs_type1.txt"), ("moscowach", "outputs_type2.txt")):
    selected = [row for row in rows if row.get("channel") == channel]
    blocks = []
    for index, row in enumerate(selected, 1):
        blocks.append(f"=== INPUT {index} ===\n{row['source_text']}\n\n=== OUTPUT {index} ===\n{row['response']}\n")
    (ROOT / "submission_v33" / filename).write_text("\n".join(blocks), encoding="utf-8")
    print(filename, len(selected))
