"""
ycb_supervisor.py
-----------------
Webots Supervisor Controller：每次模擬開始時隨機挑選 YCB 物件並排列在場景中。

使用方式：
  1. 在 .wbt 中將場景裡的 Solid 物件全部移除（只保留地板、機器手臂等）
  2. 新增一個空的 Robot 節點，設定：
       supervisor TRUE
       controller "ycb_supervisor"
  3. 將此檔案放在對應的 controllers/ycb_supervisor/ 資料夾下
"""

# from controller import Supervisor
# import random
# import math
# from config import (
#     NUM_OBJECTS, GRID_COLS, SPACING, SPAWN_HEIGHT, X_OFFSET,
#     ASSET_BASE, TARGET_OBJECTS, MASS_TABLE, ALL_OBJECTS,
#     DEFAULT_SHAPE, SHAPE_TABLE,
# )

# def make_bounding_object(name: str) -> str:
#     """根據 SHAPE_TABLE 產生對應的 boundingObject VRML 字串。"""
#     shape, dims = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
#     if shape == "Box":
#         x, y, z = dims
#         return f"boundingObject Box {{\n    size {x} {y} {z}\n  }}"
#     elif shape == "Sphere":
#         r = dims[0]
#         return f"boundingObject Sphere {{\n    radius {r}\n  }}"
#     elif shape == "Cylinder":
#         r, h = dims
#         return f"boundingObject Cylinder {{\n    radius {r}\n    height {h}\n  }}"
#     else:
#         return f"boundingObject Box {{\n    size 0.1 0.1 0.1\n  }}"


# def make_vrml(name: str, x: float, y: float, z: float) -> str:
#     try:
#         mass = MASS_TABLE[name]
#     except KeyError:
#         raise KeyError(f"{name} not found in MASS_TABLE")
#     base = f"{ASSET_BASE}/{name}/google_16k"
#     bounding = make_bounding_object(name)
#     return f"""Solid {{
#   translation {x:.4f} {y:.4f} {z:.4f} 
#   children [
#     Shape {{
#       appearance PBRAppearance {{
#         baseColorMap ImageTexture {{ url [ "{base}/texture_map.png" ] }}
#         roughness 1
#         metalness 0
#       }}
#       geometry Mesh {{ url [ "{base}/textured.obj" ] }}
#     }}
#   ]
#   name "{name}"
#   {bounding}
#   physics Physics {{ mass {mass} }}
# }}"""

from controller import Supervisor
import random
import math
import json
import os

from config import (
    NUM_OBJECTS, GRID_COLS, SPACING, SPAWN_HEIGHT, X_OFFSET, Z_OFFSET,
    ASSET_BASE, TARGET_OBJECTS, MASS_TABLE, ALL_OBJECTS,
    DEFAULT_SHAPE, SHAPE_TABLE,
)

SPAWN_CLEARANCE = 0.01
SPACING_MARGIN = 0.02
ARM_SETTLE_TIME_SEC = 2.5

# ── 1. 載入 JSON 資料 ──
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "ycb_geometries.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        YCB_GEO_DATA = json.load(f)
except:
    YCB_GEO_DATA = {}

def get_geometry(name: str):
    return YCB_GEO_DATA.get(name, {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "size": {"x": 0.1, "y": 0.1, "z": 0.1}
    })

def get_collision_half_height(name: str) -> float:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    size = get_geometry(name)["size"]
    sx, sy, sz = size["x"], size["y"], size["z"]

    if shape == "Sphere":
        return (sx + sy + sz) / 6.0
    return sz / 2.0

def get_collision_footprint(name: str) -> float:
    size = get_geometry(name)["size"]
    return max(size["x"], size["y"])

# ── 2. 修改後的 Bounding Object 產生器 ──
def make_bounding_object(name: str, sx: float, sy: float, sz: float) -> str:
    """
    根據 SHAPE_TABLE 決定形狀，但尺寸由 JSON 提供的 sx, sy, sz 決定。
    """
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)

    if shape == "Sphere":
        r = (sx + sy + sz) / 6.0
        return f"boundingObject Sphere {{ radius {r:.6f} }}"
    elif shape == "Cylinder":
        r = (sx + sy) / 4.0
        h = sz
        return f"boundingObject Cylinder {{ radius {r:.6f} height {h:.6f} }}"
    else:
        # 預設為 Box
        return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz:.6f} }}"

# ── 3. 核心 VRML 產生器 (加上中心點修正) ──
def make_vrml(name: str, x: float, y: float, z: float) -> str:
    if name not in MASS_TABLE:
        raise KeyError(f"{name} not found in MASS_TABLE")
    
    mass = MASS_TABLE[name]
    base = f"{ASSET_BASE}/{name}/google_16k"
    
    # 從 JSON 抓取幾何中心 (cx, cy, cz) 與 尺寸 (sx, sy, sz)
    geo = get_geometry(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    sx, sy, sz = geo["size"]["x"], geo["size"]["y"], geo["size"]["z"]

    # 產生碰撞形狀
    bounding = make_bounding_object(name, sx, sy, sz)

    # 對齊邏輯：
    # Transform translation {-cx} {-cy} {-cz} 會把 Mesh 移回 Solid 的原點
    return f"""Solid {{
  translation {x:.6f} {y:.6f} {z:.6f} 
  children [
    Transform {{
      translation {-cx:.6f} {-cy:.6f} {-cz:.6f}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColorMap ImageTexture {{ url [ "{base}/texture_map.png" ] }}
            roughness 1
            metalness 0
          }}
          geometry Mesh {{ url [ "{base}/textured.obj" ] }}
        }}
      ]
    }}
  ]
  name "{name}"
  {bounding}
  physics Physics {{
    density -1
    mass {mass}
  }}
}}"""

def compute_grid_positions(n: int, cols: int, spacing: float):
    """計算 n 個物件的網格座標，以原點為中心。"""
    rows = math.ceil(n / cols)
    positions = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col - (cols - 1) / 2.0) * spacing
        y = (row - (rows - 1) / 2.0) * spacing
        positions.append((x, y))
    return positions


def clear_ycb_objects(supervisor: Supervisor):
    """移除場景中所有名稱在 ALL_OBJECTS 清單內的 Solid 節點。"""
    root_children = supervisor.getRoot().getField('children')
    # 從後往前刪，避免 index 錯位
    i = root_children.getCount() - 1
    while i >= 0:
        node = root_children.getMFNode(i)
        if node is not None:
            name_field = node.getField('name')
            if name_field is not None:
                node_name = name_field.getSFString()
                if node_name in ALL_OBJECTS:
                    node.remove()
        i -= 1


def spawn_objects(supervisor: Supervisor, object_list: list):
    """根據傳入的清單生成物件到場景中。"""
    if not object_list:
        print("[Supervisor] Warning: Object list is empty.")
        return

    largest_footprint = max(get_collision_footprint(name) for name in object_list)
    safe_spacing = max(SPACING, largest_footprint + SPACING_MARGIN)
    positions = compute_grid_positions(len(object_list), GRID_COLS, safe_spacing)
    root_children = supervisor.getRoot().getField('children')
    
    for name, (grid_x, grid_y) in zip(object_list, positions):
        if name not in MASS_TABLE:
            continue
            
        final_x = grid_x + X_OFFSET 
        final_y = grid_y + Z_OFFSET
        
        safe_spawn_height = max(SPAWN_HEIGHT, get_collision_half_height(name) + SPAWN_CLEARANCE)
        vrml = make_vrml(name, final_x, final_y, safe_spawn_height)
        root_children.importMFNodeFromString(-1, vrml)


def wait_for_arm_settle(supervisor: Supervisor, timestep: int, seconds: float):
    """
    讓場景先跑一小段時間，給手臂控制器機會把機械臂移到目標姿態後再生成物件。
    """
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


# ── 主程式 ──────────────────────────────────────────────────
def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # 決定要生成的清單
    if TARGET_OBJECTS:
        current_list = TARGET_OBJECTS
        mode_text = "Specific Mode"
    else:
        # 如果沒指定，就從 ALL_OBJECTS 隨機抽
        current_list = random.sample(ALL_OBJECTS, k=min(NUM_OBJECTS, len(ALL_OBJECTS)))
        mode_text = "Random Mode"

    print(f"[Supervisor] Mode: {mode_text}")
    print("[Supervisor] Clearing existing YCB objects...")
    clear_ycb_objects(supervisor)
    print(f"[Supervisor] Waiting {ARM_SETTLE_TIME_SEC:.1f}s for arm to reach target pose...")
    if not wait_for_arm_settle(supervisor, timestep, ARM_SETTLE_TIME_SEC):
        return

    spawn_objects(supervisor, current_list)

    keyboard = supervisor.getKeyboard()
    keyboard.enable(timestep)

    while supervisor.step(timestep) != -1:
        key = keyboard.getKey()
        if key == ord('N'):
            print("[Supervisor] Resetting scene...")
            clear_ycb_objects(supervisor)
            print(f"[Supervisor] Waiting {ARM_SETTLE_TIME_SEC:.1f}s for arm to reach target pose...")
            if not wait_for_arm_settle(supervisor, timestep, ARM_SETTLE_TIME_SEC):
                break
            
            # 如果是隨機模式，重按 R 會換一批；指定模式則原樣重放
            if not TARGET_OBJECTS:
                current_list = random.sample(ALL_OBJECTS, k=min(NUM_OBJECTS, len(ALL_OBJECTS)))
            
            spawn_objects(supervisor, current_list)

if __name__ == "__main__":
    main()
