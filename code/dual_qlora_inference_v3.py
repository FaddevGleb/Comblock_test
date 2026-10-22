"""Versioned inference using style_runtime_v4 prompt ordering."""

import style_runtime_v4 as _runtime_v4
import dual_qlora_inference_v2 as _inference

_inference.runtime = _runtime_v4


def main() -> None:
    _inference.main()


if __name__ == "__main__":
    main()
