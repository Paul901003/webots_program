"""joint_teleport_probe — 測 Supervisor 能否取到 UR5e proto 內部 HingeJoint 並 setJointPosition。"""
from controller import Supervisor, Node

sv = Supervisor()
ts = int(sv.getBasicTimeStep())
ur = sv.getFromDef("UR5E")
print(f"[probe] UR5E node = {ur}, type = {ur.getTypeName() if ur else None}", flush=True)

# 遞迴走訪，收集 HingeJoint 節點
joints = []

def walk(node, depth=0):
    if node is None or depth > 40:
        return
    tn = None
    try:
        tn = node.getTypeName()
    except Exception:
        pass
    if tn == "HingeJoint":
        joints.append(node)
    # 走 children(MFNode) 與 endpoint(SFNode)
    for fname in ("children", "endpoint"):
        f = node.getField(fname)
        if f is None:
            continue
        ft = f.getType()
        if ft == Node.__dict__.get("NO_FIELD", -1):
            continue
        try:
            if fname == "children":
                for i in range(f.getCount()):
                    walk(f.getMFNode(i), depth + 1)
            else:
                walk(f.getSFNode(), depth + 1)
        except Exception as e:
            print(f"[probe]   走訪 {fname} 失敗: {e}", flush=True)

walk(ur)
print(f"[probe] 找到 HingeJoint 數: {len(joints)}", flush=True)

if joints:
    target_deg = [43.7584, -72.4674, 112.6268, -132.3083, -149.9767, 87.5146]
    import math
    for j, deg in zip(joints, target_deg):
        try:
            j.setJointPosition(math.radians(deg))
            print(f"[probe]   setJointPosition OK: {math.radians(deg):.3f} rad", flush=True)
        except Exception as e:
            print(f"[probe]   setJointPosition 失敗: {e}", flush=True)
    sv.step(ts)
    print("[probe] ★ 瞬移完成，setJointPosition 可用於此 proto", flush=True)
else:
    print("[probe] ★ 走訪取不到內部 HingeJoint（proto 封裝）→ 需改用其他方式", flush=True)

sv.step(ts)
sv.simulationQuit(0)
