from __future__ import annotations

import argparse
import json
from pathlib import Path

from k2_region_lab.krea_control_lora import inspect_krea_control_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    args = parser.parse_args()
    report = inspect_krea_control_checkpoint(
        args.checkpoint,
        allow_unverified_legacy=args.allow_unverified_legacy,
    )
    print(json.dumps(report.document(), indent=2, sort_keys=True))
    raise SystemExit(0 if report.compatible else 2)


if __name__ == "__main__":
    main()
