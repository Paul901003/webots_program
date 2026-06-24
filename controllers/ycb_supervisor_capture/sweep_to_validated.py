#!/usr/bin/env python3
"""sweep_to_validated.py — 把 generate_sweep_viewpoints 的有序軌跡轉成
A-2b / A-4 驗證器吃的 validated_viewpoints_latest.json 格式。

讀  data/viewpoints/sweep_viewpoints_latest.json
寫  data/viewpoints/validated_viewpoints_latest.json
     {"validated":[{id, joint_deg, radius_m, order, elevation_deg, azimuth_deg, ...}]}
保留 order/仰角/方位 供追溯;id 為可讀字串(驗證器以字串列印)。

用法: /usr/bin/python3 sweep_to_validated.py
"""

import json
import os

import generate_candidate_viewpoints as G

VP = G.DATA_VIEWPOINTS
SRC = os.path.join(VP, "sweep_viewpoints_latest.json")
DST = os.path.join(VP, "validated_viewpoints_latest.json")


def main():
    with open(SRC, encoding="utf-8") as f:
        d = json.load(f)

    validated = []
    for v in d["viewpoints"]:
        az = v["azimuth_deg"]
        az_tag = "zenith" if az is None else f"az{int(az):03d}"
        validated.append({
            "id": f"vp{v['order']:02d}_el{int(v['elevation_deg']):02d}_{az_tag}",
            "joint_deg": v["joint_deg"],
            "radius_m": v["radius_m"],
            "order": v["order"],
            "elevation_deg": v["elevation_deg"],
            "azimuth_deg": az,
            "camera_position_m": v["camera_position_m"],
        })

    out = {
        "source": os.path.basename(SRC),
        "center_m": d["center_m"],
        "cam_r_m": d["cam_r_m"],
        "ws_radius_m": d["ws_radius_m"],
        "note": "驗證器執行時請帶 --x-offset 0.15 --ws-offset 0.295(= cam_r - ws_r)",
        "validated": validated,
    }
    os.makedirs(VP, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"轉換 {len(validated)} 視角 → {DST}")
    print(f"  中心 {d['center_m']}  cam_r {d['cam_r_m']}  ws_r {d['ws_radius_m']}")


if __name__ == "__main__":
    main()
