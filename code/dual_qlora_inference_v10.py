"""v26 inference: strict v18 review with v26 few-shot style prompts."""

from __future__ import annotations

import dual_qlora_inference_v9 as previous
import dual_qlora_inference_v4 as implementation
import news_style_runtime_v26 as runtime


implementation.runtime = runtime
previous.implementation.runtime = runtime


if __name__ == "__main__":
    previous.main()
