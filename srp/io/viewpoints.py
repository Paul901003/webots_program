#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""viewpoints.py — 讀 A-3 挑選結果 → view 名集合(Stage 1/2 共用,單一來源)。

視角挑選一律走 A-3(select_counts.py:天頂→中心→群組 A/B→最遠點補滿),
下游只讀其產出的 selected_viewpoints_multi_n{N}_{x_tag}.json,**不在此另訂挑選邏輯**。
view 名用 el/az 還原成與拍攝一致的命名(el90→view_el90;否則 view_el{el}_az{az%360})。
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEWPOINTS_DIR = REPO / "data" / "viewpoints"
X_TAG = os.environ.get("SELECT_X_TAG", "x+035")   # A-3 挑選檔的 x_offset 標籤


def selected_view_names(num_views, x_tag=None):
    """回傳 A-3 selected_n{N} 挑的 view 名集合(set)。找不到檔就報錯提示先跑 select_counts。"""
    x_tag = x_tag or X_TAG
    f = VIEWPOINTS_DIR / f"selected_viewpoints_multi_n{num_views}_{x_tag}.json"
    if not f.is_file():
        raise FileNotFoundError(
            f"找不到 A-3 挑選檔 {f.name};請先跑 "
            f"select_counts.py --multi --x-offset 0.35 --only-latest --counts {num_views}")
    want = set()
    for v in json.loads(f.read_text(encoding="utf-8"))["selected"]:
        m = v.get("meta", {})
        el = round(abs(float(m.get("elevation_deg", 90))))
        want.add("view_el90" if el == 90
                 else f"view_el{el}_az{round(float(m.get('azimuth_deg', 0))) % 360}")
    return want
