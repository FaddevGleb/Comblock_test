"""v27 worker: DoRA over every linear module, with v26 few-shot masks."""

from __future__ import annotations

import json
from peft import LoraConfig

import dual_qlora_worker_v14 as previous
import news_style_runtime_v26 as runtime


worker = previous.worker
previous.runtime = runtime
worker.runtime = runtime

ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_dora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


previous.lora_config = lora_config
worker.lora_config = lora_config
worker.runtime = runtime
worker.ATTENTION_MODULES = ATTENTION_MODULES


def train_final(*args, **kwargs):
    """Run the inherited final training and add an auditable DoRA config."""
    previous.train_final(*args, **kwargs)
    channel = kwargs.get("channel", args[1] if len(args) > 1 else None)
    root = kwargs.get("root", args[3] if len(args) > 3 else None)
    if channel is None or root is None:
        return
    config_path = root.parent.parent / "adapters" / channel / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "adapter_method": "DoRA",
        "use_dora": True,
        "target_modules": "all-linear",
        "attention_modules": list(ATTENTION_MODULES),
        "mlp_modules_included": True,
        "few_shot_style_mask": True,
    })
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


worker.train_final = train_final


if __name__ == "__main__":
    worker.main()
