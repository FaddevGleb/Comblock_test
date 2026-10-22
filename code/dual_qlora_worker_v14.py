"""v26 worker: v19 training settings with few-shot masks and attention audit."""

from __future__ import annotations

from peft import LoraConfig

import dual_qlora_worker_v13 as previous
import news_style_runtime_v26 as runtime


worker = previous.worker
previous.runtime = runtime
worker.runtime = runtime

ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_config() -> LoraConfig:
    # PEFT's all-linear selector already includes these attention projections
    # and the MLP projections. Keep the selector, and record the explicit
    # attention coverage for the run audit.
    return LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, use_rslora=True,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )


previous.lora_config = lora_config
worker.lora_config = lora_config
worker.runtime = runtime
worker.ATTENTION_MODULES = ATTENTION_MODULES


if __name__ == "__main__":
    worker.main()
