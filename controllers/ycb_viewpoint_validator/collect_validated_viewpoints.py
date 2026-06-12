#!/usr/bin/env python3
"""A-3b: 收集所有通過驗證的視角，寫入 selected_viewpoints.json（不篩選）。

單半徑（預設）：
  python collect_validated_viewpoints.py

多半徑：
  python collect_validated_viewpoints.py --multi
  python collect_validated_viewpoints.py \
      --input data/viewpoints/validated_viewpoints_multi.json \
      --output data/viewpoints/selected_viewpoints_multi.json
"""

import argparse
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
VIEWPOINTS_DIR = REPO_ROOT / "data" / "viewpoints"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multi", action="store_true",
                        help="多半徑模式（等同 --input validated_viewpoints_multi.json "
                             "--output selected_viewpoints_multi.json）")
    parser.add_argument("--input", type=Path, default=None,
                        help="輸入 validated_viewpoints JSON 路徑")
    parser.add_argument("--output", type=Path, default=None,
                        help="輸出 selected_viewpoints JSON 路徑")
    args = parser.parse_args()

    if args.multi:
        validated_path = VIEWPOINTS_DIR / "validated_viewpoints_multi_latest.json"
        selected_path  = VIEWPOINTS_DIR / "selected_viewpoints_multi.json"
    else:
        validated_path = VIEWPOINTS_DIR / "validated_viewpoints_latest.json"
        selected_path  = VIEWPOINTS_DIR / "selected_viewpoints.json"

    if args.input:
        validated_path = args.input
    if args.output:
        selected_path = args.output

    data = json.loads(validated_path.read_text(encoding="utf-8"))
    validated = data.get("validated", [])

    radii = sorted({vp.get("meta", {}).get("radius_m") or vp.get("radius_m")
                    for vp in validated} - {None})

    result = {
        "source": str(validated_path),
        "validated_count": len(validated),
        "selected_count": len(validated),
        "radii_m": radii,
        "selected": validated,
    }
    selected_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"收集 {len(validated)} 個已驗證視角")
    if radii:
        for r in radii:
            count = sum(1 for vp in validated
                        if (vp.get("meta", {}).get("radius_m") or vp.get("radius_m")) == r)
            print(f"  r={r}m: {count} 個")
    print(f"輸出: {selected_path}")


if __name__ == "__main__":
    main()
