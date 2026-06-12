"""
ros2_bridge_utils.py

ROS2 bridge subprocess 工具函式，供 ycb_supervisor_ros2_test 和
ycb_viewpoint_validator 共用。不依賴 Webots controller 模組。
"""

import json
import os
import queue
import subprocess
import threading
import time

BRIDGE_PYTHON = "/usr/bin/python3.12"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_SCRIPT = os.path.join(_THIS_DIR, "ros2_bridge_subprocess.py")


def launch_ros2_bridge():
    """啟動 bridge 子行程，立即返回。用執行緒在背景讀取 stdout，避免阻塞。"""
    if not os.path.isfile(BRIDGE_PYTHON):
        print(f"[Supervisor] 找不到 {BRIDGE_PYTHON}，改用直接移動模式")
        return None, None
    try:
        proc = subprocess.Popen(
            [BRIDGE_PYTHON, BRIDGE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        print(f"[Supervisor] 無法啟動 ROS2 bridge: {e}")
        return None, None

    line_queue = queue.Queue()

    def _reader():
        for line in proc.stdout:
            line_queue.put(line.rstrip("\n"))
        line_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    print("[Supervisor] ROS2 bridge 子行程已啟動")
    return proc, line_queue


def wait_for_bridge_ready(supervisor, timestep: int, proc, line_queue, timeout_sec: float):
    """在繼續 supervisor.step() 的同時，等待 bridge 的 READY 訊號。"""
    if proc is None:
        return None
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if supervisor.step(timestep) == -1:
            proc.terminate()
            return None
        if proc.poll() is not None:
            err = proc.stderr.read()
            print(f"[Supervisor] ROS2 bridge 意外結束:\n{err}")
            return None
        while True:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                print("[Supervisor] ROS2 bridge stdout 已關閉")
                return None
            if line == "READY":
                print("[Supervisor] ROS2 bridge 已就緒")
                return proc
            if line:
                print(f"[Bridge] {line}")
    print("[Supervisor] ROS2 bridge 啟動逾時，改用直接移動模式")
    proc.terminate()
    return None


def request_plan(proc, line_queue: queue.Queue,
                 current_joints: list, target_joints: list, collision_objects: list,
                 timeout: float = 60.0, supervisor=None, timestep: int = 0):
    """透過 subprocess 向 planning_bridge 請求 MoveIt 規劃。
    結果從 line_queue 讀取，避免與背景讀取執行緒競爭 proc.stdout。
    若傳入 supervisor 與 timestep，等待期間持續呼叫 supervisor.step()，避免 sim 時間凍結。
    """
    request = json.dumps({
        "current_joints": [float(v) for v in current_joints],
        "target_joints": [float(v) for v in target_joints],
        "collision_objects": collision_objects,
    })
    try:
        proc.stdin.write(request + "\n")
        proc.stdin.flush()
    except Exception as e:
        print(f"[Supervisor] 送出規劃請求失敗: {e}")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        if supervisor is not None and timestep > 0:
            if supervisor.step(timestep) == -1:
                return None

        try:
            line = line_queue.get(timeout=0.0 if (supervisor is not None and timestep > 0) else min(1.0, deadline - time.time()))
        except queue.Empty:
            continue
        if line is None:
            print("[Supervisor] ROS2 bridge stdout 已關閉")
            return None
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"[Bridge] {line}")  # 非 JSON 行（例如 debug log）繼續等
    print("[Supervisor] 等待規劃結果逾時")
    return None


def stop_ros2_bridge(proc):
    if proc is None:
        return
    try:
        proc.stdin.close()
        proc.wait(timeout=3.0)
    except Exception:
        proc.terminate()
