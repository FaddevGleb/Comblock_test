"""v32 worker: correctly inject DoRA r=8 into the inherited train functions."""

from __future__ import annotations

import json

from peft import LoraConfig

import dual_qlora_worker_v14 as previous
import news_style_runtime_v26 as runtime


worker = previous.worker
previous.runtime = runtime
worker.runtime = runtime
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_BASE_TRAIN_FINAL = worker.train_final


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_dora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def train_final(*args, **kwargs):
    _BASE_TRAIN_FINAL(*args, **kwargs)
    channel = kwargs.get("channel", args[1] if len(args) > 1 else None)
    root = kwargs.get("root", args[3] if len(args) > 3 else None)
    if channel is None or root is None:
        return
    config_path = root.parent.parent / "adapters" / channel / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("r") != 8 or config.get("lora_alpha") != 16 or not config.get("use_dora"):
        raise RuntimeError(f"Wrong adapter config for {channel}: {config}")
    metadata = {
        "channel": channel,
        "rank": 8,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "adapter_method": "DoRA",
        "use_dora": True,
        "attention_modules": list(ATTENTION_MODULES),
        "mlp_modules_included": True,
        "few_shot_style_mask": True,
    }
    (config_path.parent / "training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# train_split/train_final are functions defined in v13 and resolve lora_config
# in v13's globals. Patch that exact module as well as the wrapper and worker.
previous.previous.lora_config = lora_config
previous.lora_config = lora_config
worker.lora_config = lora_config
worker.runtime = runtime
worker.ATTENTION_MODULES = ATTENTION_MODULES
worker.train_final = train_final


if __name__ == "__main__":
    worker.main()
