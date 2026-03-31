import os
import json
from functools import lru_cache
from typing import Dict, Any

from config import ALL_OBJECTS
ASSET_BASE = "../../urdfs/ycb_assets"



# ── OBJ 幾何解析 ────────────────────────────────────────────

def compute_obj_geometry(obj_path: str) -> Dict[str, Any]:
    """解析 OBJ 檔案，回傳幾何中心與各軸尺寸。"""
    if not os.path.isfile(obj_path):
        return None

    xs, ys, zs = [], [], []
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                xs.append(float(parts[1]))
                ys.append(float(parts[2]))
                zs.append(float(parts[3]))
    except Exception as e:
        print(f"Error parsing {obj_path}: {e}")
        return None

    if not xs:
        return None

    # 計算 Bounding Box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    return {
        "center": {
            "x": (max_x + min_x) / 2.0,
            "y": (max_y + min_y) / 2.0,
            "z": (max_z + min_z) / 2.0
        },
        "size": {
            "x": max_x - min_x,
            "y": max_y - min_y,
            "z": max_z - min_z
        }
    }

# ── 執行掃描 ────────────────────────────────────────────────

def main():
    results = {}
    print(f"{'Object Name':<30} | {'Size (X, Y, Z)':<30}")
    print("-" * 65)

    for name in ALL_OBJECTS:
        obj_path = os.path.join(ASSET_BASE, name, "google_16k", "textured.obj")
        
        geo = compute_obj_geometry(obj_path)
        
        if geo:
            results[name] = geo
            size = geo["size"]
            size_str = f"{size['x']:.4f}, {size['y']:.4f}, {size['z']:.4f}"
            print(f"{name:<30} | {size_str}")
        else:
            print(f"{name:<30} | [FAILED] File not found or empty")

    # 輸出成 JSON 檔案供後續 Webots 腳本快速讀取
    output_file = "ycb_geometries.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print("-" * 65)
    print(f"Scan complete. Data saved to {output_file}")

if __name__ == "__main__":
    main()