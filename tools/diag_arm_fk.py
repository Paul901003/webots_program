#!/usr/bin/env python3
"""diag_arm_fk.py — 比對 generate_labels 的手臂 FK vs 已驗證正確的 FK,定位手臂畫錯位置的原因。

比對:
  A = generate_labels._ur5e_link_transforms(q)["wrist_3"]          (手臂渲染用的 FK)
  B = generate_candidate_viewpoints.webots_tool_slot_transform_world(q)
      去掉末端 tool-slot 平移後的 wrist_3 frame                     (已驗證:相機位姿=實拍)
若 A≈B → FK 相同且正確 → 手臂畫錯是「mesh 原點/朝向」問題(下一步查 mesh)。
若 A≠B → FK 本身有誤 → 修 _ur5e_link_transforms。

用法(webots_visual_hull 環境):
  /home/cho/.pyenv/versions/webots_visual_hull/bin/python3 tools/diag_arm_fk.py
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
GCAP = os.path.join(REPO, "controllers", "ycb_supervisor_capture")
sys.path.insert(0, GCAP)
# ur_config 在 my_ur_kinematics
sys.path.insert(0, os.path.join(REPO, "controllers", "ur5e_controller", "my_ur_kinematics"))
sys.path.insert(0, os.path.join(REPO, "controllers", "ur5e_controller"))

import generate_labels as GL
import generate_candidate_viewpoints as GV

# 幾組測試關節角(取自 validated)
TESTS = [
    [-15.1, -88.0, 77.1, -93.6, -86.2, -14.7],
    [-44.5, -98.7, 98.1, -98.1, -61.2, -72.1],
    [43.7584, -72.4674, 112.6268, -132.3083, -149.9767, 87.5146],
]

print(f"{'joint_deg':<48}{'|Δpos| mm':>12}{'ΔR max':>10}")
for jd in TESTS:
    q = [math.radians(d) for d in jd]
    A = GL._ur5e_link_transforms(q)["wrist_3"]                 # generate_labels wrist_3
    # 已驗證 FK 的 tool-slot transform,退掉末端 tool-slot 平移 → wrist_3 frame
    T_tool = GV.webots_tool_slot_transform_world(q)
    tool_slot_t = np.array(GV.WEBOTS_TOOL_SLOT_TRANSLATION_M, float)
    B = T_tool.copy()
    B[:3, 3] = B[:3, 3] - B[:3, :3] @ tool_slot_t              # 移除末端平移
    dpos = np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000
    dR = np.max(np.abs(A[:3, :3] - B[:3, :3]))
    print(f"{str([round(x,1) for x in jd]):<48}{dpos:>12.3f}{dR:>10.5f}")

print("\n判讀:|Δpos|<1mm 且 ΔR<0.001 → 兩 FK 相同(FK 正確)→ 手臂畫錯在 mesh;否則 FK 有誤。")
# 附:也印出 base/shoulder 等中間連桿位置(供進一步定位)
q0 = [math.radians(d) for d in TESTS[0]]
tf = GL._ur5e_link_transforms(q0)
print("\ngenerate_labels 連桿世界位置(第一組 joint):")
for k in ("base", "shoulder", "upper", "forearm", "wrist_1", "wrist_2", "wrist_3"):
    print(f"  {k:9s} {np.round(tf[k][:3,3],4).tolist()}")
