#!/usr/bin/python3
"""gen_multicam_world.py — 由驗證視角生成「多相機同步拍攝」世界檔。

讀 data/viewpoints/<validated>.json(預設 validated_viewpoints_multi_latest.json),
每個路徑可行視角(planning.success)在 ray_origin_m 擺一台靜態 IntelRealsenseD455,
光軸(local +X)對準 ray_axis_world、roll=0(up→世界+Z;天頂用 fallback +Y)。
產物:
  - worlds/ycb_multicam_capture.wbt        N 台相機 + 一個 supervisor(ycb_supervisor_multicam)
  - controllers/ycb_supervisor_multicam/multicam_viewpoints.json  side-car:每台 {def,view,id,joint_deg}
相機朝向慣例已用 12 個 selected 視角對照既有 view_XX_pose.json 驗證(rpy 誤差<0.5°)。
用法: /usr/bin/python3 gen_multicam_world.py [validated.json]
"""
import json
import math
import os
import sys

import numpy as np

CUR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(CUR))
VIEWPOINTS_DIR = os.path.join(REPO, "data", "viewpoints")
if CUR not in sys.path:
    sys.path.insert(0, CUR)
import config as CFG   # noqa: E402  視角來源檔由此決定
WORLD_OUT = os.path.join(REPO, "worlds", "ycb_multicam_capture.wbt")
SIDECAR_OUT = os.path.join(CUR, "multicam_viewpoints.json")


def R_from_axis(f, up=(0.0, 0.0, 1.0), fallback=(0.0, 1.0, 0.0)):
    """相機光軸=local +X=f;local Y 水平、local Z 朝上(roll=0)。回傳 3x3(欄=local 軸的世界方向)。"""
    f = np.asarray(f, float); f = f / np.linalg.norm(f)
    up = np.asarray(up, float)
    if np.linalg.norm(np.cross(up, f)) < 1e-6:   # 天頂:光軸≈±世界Z → 換 fallback
        up = np.asarray(fallback, float)
    ly = np.cross(up, f); ly /= np.linalg.norm(ly)
    lz = np.cross(f, ly); lz /= np.linalg.norm(lz)
    return np.column_stack([f, ly, lz])


def R_to_axis_angle(m):
    """3x3 旋轉矩陣 → Webots axis-angle [ax,ay,az,angle]。"""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    if abs(math.pi - angle) < 1e-6:                      # 180°
        d = [(m[i, i] + 1.0) / 2.0 for i in range(3)]
        i = int(np.argmax(d)); ax = [0.0, 0.0, 0.0]; ax[i] = math.sqrt(max(0.0, d[i]))
        j, k = [(i + 1) % 3, (i + 2) % 3]
        if ax[i] > 1e-9:
            ax[j] = (m[i, j] + m[j, i]) / (4.0 * ax[i])
            ax[k] = (m[i, k] + m[k, i]) / (4.0 * ax[i])
        return [ax[0], ax[1], ax[2], angle]
    d = 2.0 * math.sin(angle)
    v = [(m[2, 1] - m[1, 2]) / d, (m[0, 2] - m[2, 0]) / d, (m[1, 0] - m[0, 1]) / d]
    n = math.sqrt(sum(c * c for c in v))
    return [v[0] / n, v[1] / n, v[2] / n, angle]


CAM_TEMPLATE = """DEF {defname} IntelRealsenseD455 {{
  translation {tx:.6f} {ty:.6f} {tz:.6f}
  rotation {ax:.6f} {ay:.6f} {az:.6f} {ang:.6f}
  name "{camname}"
  controller "realsense_auto_capture_controller"
  resolution "HD"
  minRange 0.3
  maxRange 3.0
  fps 30
  physics NULL
}}"""

WORLD_HEADER = """#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "../protos/blank_floor.proto"
EXTERNPROTO "../protos/IntelRealsenseD455.proto"

WorldInfo {
  basicTimeStep 16
  defaultDamping Damping {
    linear 0.5
    angular 0.5
  }
  contactProperties [
    ContactProperties {
      coulombFriction [ 1 ]
      bounce 0
      bounceVelocity 0
      softCFM 0.0001
    }
  ]
}
Viewpoint {
  orientation 1 0 -1 3.14
  position 0 0 5.5
  followType "None"
}
TexturedBackground {
}
TexturedBackgroundLight {
  luminosity 1.2
  castShadows FALSE
}
blank_floor {
}
Robot {
  supervisor TRUE
  controller "ycb_supervisor_multicam"
  controllerArgs []
  customData "capture_mode=multi"
  children [
  ]
}
"""


# FK:從 joint_deg 算相機「完整」旋轉(含 roll),修正天頂 roll 退化;非天頂與 ray_axis+roll=0 等價。
_GCAP_DIR = os.path.join(REPO, "controllers", "ycb_supervisor_capture")
if _GCAP_DIR not in sys.path:
    sys.path.insert(0, _GCAP_DIR)
try:
    from generate_candidate_viewpoints import webots_camera_transform_world as _fk_cam
except Exception as _e:   # noqa: BLE001
    _fk_cam = None
    print(f"[gen] 警告:FK 不可用({_e}),旋轉退回 ray_axis+roll=0")


def camera_R_world(joint_deg, axis):
    """相機完整旋轉(欄=相機 local 軸的世界方向,X=光軸):
    有 joint_deg → 用 FK(含正確 roll);否則 ray_axis + roll=0。"""
    if _fk_cam is not None and joint_deg:
        return _fk_cam([math.radians(d) for d in joint_deg])[:3, :3]
    return R_from_axis(axis)


def el_az_name(origin, target):
    """相機位置相對物體中心的 el/az → 檔名(如 el60_az135;天頂 el90)。兩邊一致。"""
    d = np.asarray(origin, float) - np.asarray(target, float)
    dist = float(np.linalg.norm(d))
    el = round(math.degrees(math.asin(max(-1.0, min(1.0, d[2] / max(dist, 1e-9))))))
    if el >= 88:
        return "view_el90"
    az = round(math.degrees(math.atan2(d[1], d[0])) % 360)
    return f"view_el{el:02d}_az{az:03d}"


def load_viewpoints(path):
    d = json.load(open(path, encoding="utf-8"))
    vps = d.get("validated") or d.get("selected") or d.get("viewpoints") or []
    out = []
    for vp in vps:
        if not vp.get("ok", True):
            continue
        if vp.get("planning") and not vp["planning"].get("success", True):
            continue
        ray = vp.get("ray") or {}
        org = ray.get("ray_origin_m"); axis = ray.get("ray_axis_world")
        if not org or not axis:
            continue
        out.append({"id": vp.get("id"), "origin": org, "axis": axis,
                    "joint_deg": vp.get("joint_deg")})
    return out, d


def main():
    # 視角來源:argv 優先,否則用 config.py 指定的檔(決定相機數量與位姿)。
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VIEWPOINTS_DIR, CFG.VIEWPOINTS_FILE)
    print(f"[gen] 視角來源(config): {os.path.basename(src)}")
    vps, meta = load_viewpoints(src)
    if not vps:
        sys.exit(f"[gen] 無可用視角: {src}")

    target = meta.get("target_m") or [0.35, 0.0, 0.0]
    cam_blocks = []
    sidecar = []
    for i, vp in enumerate(vps):
        defname = f"CAM_{i:02d}"
        camname = f"cam_{i:02d}"
        view = el_az_name(vp["origin"], target)   # 檔名 = el/az(兩邊一致對應)
        R = camera_R_world(vp.get("joint_deg"), vp["axis"])   # FK 完整旋轉(含 roll)
        aa = R_to_axis_angle(R)
        # 節點(body)放在 ray_origin − R·[0.005,0,0],讓 D455 感測器(沿相機+X 偏 5mm)
        # 正好落在 ray_origin,與手臂版(感測器在 ray_origin)位姿完全一致。
        node = np.asarray(vp["origin"], float) - R[:, 0] * 0.005
        ox, oy, oz = node
        cam_blocks.append(CAM_TEMPLATE.format(
            defname=defname, camname=camname, tx=ox, ty=oy, tz=oz,
            ax=aa[0], ay=aa[1], az=aa[2], ang=aa[3]))
        sidecar.append({"def": defname, "view": view, "id": vp["id"],
                        "joint_deg": vp["joint_deg"], "origin_m": vp["origin"],
                        "axis_world": vp["axis"]})

    world = WORLD_HEADER + "\n" + "\n".join(cam_blocks) + "\n"
    with open(WORLD_OUT, "w", encoding="utf-8") as f:
        f.write(world)
    json.dump({"source": os.path.basename(src), "target_m": meta.get("target_m"),
               "count": len(sidecar), "cameras": sidecar},
              open(SIDECAR_OUT, "w", encoding="utf-8"), indent=2)
    print(f"[gen] {len(sidecar)} 台相機 → {WORLD_OUT}")
    print(f"[gen] side-car → {SIDECAR_OUT}")


if __name__ == "__main__":
    main()
