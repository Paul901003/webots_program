from controller import Supervisor
import math

from config import (
    CAPTURE_COLOR_RGB,
    CAPTURE_CENTER_X,
    CAPTURE_CENTER_Y,
    CAPTURE_CENTER_Z,
    CAPTURE_RADIUS,
    CAPTURE_WORKSPACE_DEF,
    CAPTURE_WORKSPACE_NAME,
    COLOR_RGB,
    LATITUDE_BANDS,
    LONGITUDE_BANDS,
    ROBOT_BASE_Z,
    ROBOT_BASE_X,
    ROBOT_BASE_Y,
    SEGMENTS_PER_RING,
    TABLE_HEIGHT_Z,
    TRANSPARENCY,
    UR5E_DEF,
    WORKSPACE_CENTER_Z,
    WORKSPACE_DEF,
    WORKSPACE_NAME,
    WORKSPACE_RADIUS,
)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def make_polyline(points):
    coord_lines = []
    index_tokens = []
    for x, y, z in points:
        coord_lines.append(f"            {x:.6f} {y:.6f} {z:.6f}")
    for idx in range(len(points)):
        index_tokens.append(str(idx))
    index_tokens.append("-1")
    return "\n".join(coord_lines), " ".join(index_tokens)


def make_line_shape(points, color_rgb, transparency):
    coord_text, index_text = make_polyline(points)
    r, g, b = color_rgb
    return f"""    Shape {{
      appearance Appearance {{
        material Material {{
          diffuseColor {r:.3f} {g:.3f} {b:.3f}
          emissiveColor {r:.3f} {g:.3f} {b:.3f}
          transparency {transparency:.3f}
        }}
      }}
      geometry IndexedLineSet {{
        coord Coordinate {{
          point [
{coord_text}
          ]
        }}
        coordIndex [ {index_text} ]
      }}
      castShadows FALSE
    }}"""


def build_ring(base_x: float, base_y: float, z_value: float, radius_xy: float):
    points = []
    for step in range(SEGMENTS_PER_RING + 1):
        theta = 2.0 * math.pi * step / SEGMENTS_PER_RING
        x = base_x + radius_xy * math.cos(theta)
        y = base_y + radius_xy * math.sin(theta)
        points.append((x, y, z_value))
    return points


def build_meridian(base_x: float, base_y: float, center_z: float, radius: float, theta: float, min_phi: float):
    points = []
    for step in range(LATITUDE_BANDS * 4 + 1):
        t = step / max(1, LATITUDE_BANDS * 4)
        phi = min_phi + (math.pi / 2.0 - min_phi) * t
        radial = radius * math.cos(phi)
        z = center_z + radius * math.sin(phi)
        x = base_x + radial * math.cos(theta)
        y = base_y + radial * math.sin(theta)
        points.append((x, y, z))
    return points


def resolve_ur5e_base_pose(supervisor: Supervisor):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    if ur5e_node is None:
        return ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z

    translation_field = ur5e_node.getField("translation")
    if translation_field is None:
        return ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z

    tx, ty, tz = translation_field.getSFVec3f()
    return tx, ty, tz


def build_hemisphere_shapes(base_x: float, base_y: float, center_z: float, table_z: float, radius: float, color_rgb):
    clipped_ratio = clamp(
        (table_z - center_z) / radius,
        -1.0,
        1.0,
    )
    min_phi = math.asin(clipped_ratio)

    shapes = []

    base_ring_radius = radius * math.cos(min_phi)
    if base_ring_radius > 1e-6:
        shapes.append(
            make_line_shape(
                build_ring(base_x, base_y, table_z, base_ring_radius),
                color_rgb,
                TRANSPARENCY,
            )
        )

    for band in range(1, LATITUDE_BANDS + 1):
        phi = min_phi + (math.pi / 2.0 - min_phi) * band / LATITUDE_BANDS
        z_value = center_z + radius * math.sin(phi)
        radius_xy = radius * math.cos(phi)
        if radius_xy > 1e-6:
            shapes.append(
                make_line_shape(
                    build_ring(base_x, base_y, z_value, radius_xy),
                    color_rgb,
                    TRANSPARENCY,
                )
            )

    for band in range(LONGITUDE_BANDS):
        theta = 2.0 * math.pi * band / LONGITUDE_BANDS
        shapes.append(
            make_line_shape(
                build_meridian(base_x, base_y, center_z, radius, theta, min_phi),
                color_rgb,
                TRANSPARENCY,
            )
        )

    return shapes


def build_workspace_node(def_name: str, node_name: str, shapes):
    children_text = "\n".join(shapes)
    return f"""DEF {def_name} Solid {{
  children [
{children_text}
  ]
  name "{node_name}"
}}"""


def remove_existing_workspace(supervisor: Supervisor):
    for def_name in (
        WORKSPACE_DEF,
        CAPTURE_WORKSPACE_DEF,
    ):
        node = supervisor.getFromDef(def_name)
        if node is not None:
            node.remove()


def main():
    supervisor = Supervisor()
    base_x, base_y, base_z = resolve_ur5e_base_pose(supervisor)
    workspace_center_z = base_z + WORKSPACE_CENTER_Z
    workspace_table_z = base_z + TABLE_HEIGHT_Z

    remove_existing_workspace(supervisor)

    supervisor.getRoot().getField("children").importMFNodeFromString(
        -1,
        build_workspace_node(
            WORKSPACE_DEF,
            WORKSPACE_NAME,
            build_hemisphere_shapes(
                base_x,
                base_y,
                workspace_center_z,
                workspace_table_z,
                WORKSPACE_RADIUS,
                COLOR_RGB,
            ),
        ),
    )
    supervisor.getRoot().getField("children").importMFNodeFromString(
        -1,
        build_workspace_node(
            CAPTURE_WORKSPACE_DEF,
            CAPTURE_WORKSPACE_NAME,
            build_hemisphere_shapes(
                CAPTURE_CENTER_X,
                CAPTURE_CENTER_Y,
                CAPTURE_CENTER_Z,
                workspace_table_z,
                CAPTURE_RADIUS,
                CAPTURE_COLOR_RGB,
            ),
        ),
    )

    timestep = int(supervisor.getBasicTimeStep())
    while supervisor.step(timestep) != -1:
        pass


if __name__ == "__main__":
    main()
