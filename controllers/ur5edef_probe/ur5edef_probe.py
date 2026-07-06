"""ur5edef_probe — 驗證真 UR5e(DEF版) setJointPosition 後、不 resetPhysics 會不會垮。"""
import math
import traceback
from controller import Supervisor

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TEST = [-15.1, -88.0, 77.1, -93.6, -86.2, -14.7]

try:
    sv = Supervisor(); ts = int(sv.getBasicTimeStep())
    arm = sv.getFromDef("ARM"); cam = sv.getFromDef("PROBE_CAM")
    jn = [arm.getFromProtoDef(n) for n in JOINTS] if arm else []
    ok = bool(jn) and all(j is not None for j in jn)
    print(f"[probe] arm={arm is not None} cam={cam is not None} joints_ok={ok}", flush=True)
    if not (arm and cam and ok):
        print("[probe] ★取節點失敗", flush=True); sv.simulationQuit(1)
    for j, d in zip(jn, TEST):
        j.setJointPosition(math.radians(d))
    print("[probe] setJointPosition 已套用,開始量測(不 resetPhysics)...", flush=True)
    sv.step(ts)
    p0 = list(cam.getPosition())
    for _ in range(120):
        if sv.step(ts) == -1:
            break
    p1 = list(cam.getPosition())
    drift = math.sqrt(sum((p1[k] - p0[k]) ** 2 for k in range(3))) * 1000
    print(f"[probe] cam t0 = {[round(v,4) for v in p0]}", flush=True)
    print(f"[probe] cam t1 = {[round(v,4) for v in p1]}", flush=True)
    print(f"[probe] 2秒相機漂移 = {drift:.2f} mm", flush=True)
    print(f"[probe] 判定: {'✓ 穩定,真UR5e setJointPosition 不需 resetPhysics' if drift < 1.0 else '✗ 會垮/漂,需 resetPhysics 或改馬達'}", flush=True)
    sv.simulationQuit(0)
except Exception:
    print("[probe] ★例外:\n" + traceback.format_exc(), flush=True)
