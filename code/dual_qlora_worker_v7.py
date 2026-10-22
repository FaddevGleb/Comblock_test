"""Versioned worker using style_runtime_v4 prompt ordering."""

import style_runtime_v4 as _runtime_v4
import dual_qlora_worker_v6 as _worker

_worker.runtime = _runtime_v4


def main() -> None:
    _worker.main()


if __name__ == "__main__":
    main()
