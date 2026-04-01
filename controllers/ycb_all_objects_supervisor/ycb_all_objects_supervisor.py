from controller import Supervisor
import json
import math
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_YCB_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "ycb_supervisor"))

if SHARED_YCB_DIR not in sys.path:
    sys.path.append(SHARED_YCB_DIR)

from config import ALL_OBJECTS, ASSET_BASE, DEFAULT_SHAPE, MASS_TABLE, SHAPE_TABLE


GRID_COLS = 8
SPACING = 0.28
SPAWN_HEIGHT = 0.03
X_OFFSET = 0.0
Z_OFFSET = 0.0
SPAWN_CLEARANCE = 0.01
SPACING_MARGIN = 0.02


def load_geometry_data() -> dict:
    json_path = os.path.join(SHARED_YCB_DIR, "ycb_geometries.json")
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


YCB_GEO_DATA = load_geometry_data()

def get_geometry(name: str) -> dict:
    return YCB_GEO_DATA.get(
        name,
        {
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "size": {"x": 0.1, "y": 0.1, "z": 0.1},
        },
    )

def get_collision_half_height(name: str) -> float:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    size = get_geometry(name)["size"]
    sx = size["x"]
    sy = size["y"]
    sz = size["z"]

    if shape == "Sphere":
        return (sx + sy + sz) / 6.0
    return sz / 2.0

def get_collision_footprint(name: str) -> float:
    size = get_geometry(name)["size"]
    return max(size["x"], size["y"])


def make_bounding_object(name: str, sx: float, sy: float, sz: float) -> str:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)

    if shape == "Sphere":
        radius = (sx + sy + sz) / 6.0
        return f"boundingObject Sphere {{ radius {radius:.6f} }}"
    if shape == "Cylinder":
        radius = (sx + sy) / 4.0
        return f"boundingObject Cylinder {{ radius {radius:.6f} height {sz:.6f} }}"
    return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz:.6f} }}"


def make_vrml(name: str, x: float, y: float, z: float) -> str:
    if name not in MASS_TABLE:
        raise KeyError(f"{name} not found in MASS_TABLE")

    base = f"{ASSET_BASE}/{name}/google_16k"
    geometry = get_geometry(name)

    cx = geometry["center"]["x"]
    cy = geometry["center"]["y"]
    cz = geometry["center"]["z"]
    sx = geometry["size"]["x"]
    sy = geometry["size"]["y"]
    sz = geometry["size"]["z"]
    bounding = make_bounding_object(name, sx, sy, sz)
    mass = MASS_TABLE[name]

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


def compute_grid_positions(total: int, cols: int, spacing: float):
    rows = math.ceil(total / cols)
    positions = []
    for index in range(total):
        col = index % cols
        row = index // cols
        x = (col - (cols - 1) / 2.0) * spacing
        y = (row - (rows - 1) / 2.0) * spacing
        positions.append((x, y))
    return positions


def clear_ycb_objects(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    index = root_children.getCount() - 1
    while index >= 0:
        node = root_children.getMFNode(index)
        if node is not None:
            name_field = node.getField("name")
            if name_field is not None and name_field.getSFString() in ALL_OBJECTS:
                node.remove()
        index -= 1


def spawn_all_objects(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    largest_footprint = max(get_collision_footprint(name) for name in ALL_OBJECTS)
    safe_spacing = max(SPACING, largest_footprint + SPACING_MARGIN)
    positions = compute_grid_positions(len(ALL_OBJECTS), GRID_COLS, safe_spacing)

    for name, (grid_x, grid_y) in zip(ALL_OBJECTS, positions):
        safe_spawn_height = max(SPAWN_HEIGHT, get_collision_half_height(name) + SPAWN_CLEARANCE)
        vrml = make_vrml(
            name,
            grid_x + X_OFFSET,
            grid_y + Z_OFFSET,
            safe_spawn_height,
        )
        root_children.importMFNodeFromString(-1, vrml)


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    print(f"[All Objects Supervisor] Spawning {len(ALL_OBJECTS)} YCB objects.")
    clear_ycb_objects(supervisor)
    spawn_all_objects(supervisor)

    keyboard = supervisor.getKeyboard()
    keyboard.enable(timestep)

    while supervisor.step(timestep) != -1:
        if keyboard.getKey() == ord("R"):
            print("[All Objects Supervisor] Respawning all YCB objects.")
            clear_ycb_objects(supervisor)
            spawn_all_objects(supervisor)


if __name__ == "__main__":
    main()
