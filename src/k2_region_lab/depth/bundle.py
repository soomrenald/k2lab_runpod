from __future__ import annotations

import argparse
import json
from pathlib import Path

from k2core.depth import (
    DepthFeatureFlags,
    depth_preview,
    load_blender_depth_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and import a Blender depth bundle.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.inspect_only and not DepthFeatureFlags.from_environment().blender_bundle_import:
        raise SystemExit("K2_BLENDER_BUNDLE_IMPORT_ENABLED is false")
    bundle = load_blender_depth_bundle(args.bundle)
    args.output.mkdir(parents=True, exist_ok=True)
    depth_preview(bundle.depth).save(args.output / "depth-preview.png")
    (args.output / "bundle.json").write_text(
        json.dumps(bundle.document(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(bundle.document(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
