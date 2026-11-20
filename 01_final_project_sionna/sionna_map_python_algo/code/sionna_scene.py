# ===== [원본과 다름] 원본에 없던 새 파일: Sionna RT용 방 장면(XML/PLY) 생성. 파일 전체가 새로 추가된 코드 =====
"""Sionna RT scene for the map (fingerprint) step.

The geometry is exactly the handout room (config.OBSTACLES); only the radio
physics is richer than the handout's LOS + four wall reflections:

* ``scene/room_rt.xml``     : 4 perimeter walls, floor, ceiling and all 12
  obstacles.  Every surface uses an ITU-R P.2040 material and therefore takes
  part in specular reflection, refraction (slab transmission), diffraction and
  diffuse scattering.
* ``scene/room_render.xml`` : the same scene without the ceiling, for figures.

Meshes are written as binary PLY so no extra package is needed.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import config

# Material per surface class: (ITU type, slab thickness per face [m], scattering coefficient).
# Sionna treats each mesh face as a slab, so a ray that crosses a closed obstacle
# passes two faces; the face thickness is therefore half the obstacle thickness.
SURFACE_MATERIALS: dict[str, tuple[str, float, float]] = {
    "room": ("concrete", 0.20, config.RT_SCATTERING_COEFFICIENT),     # perimeter walls, floor, ceiling
    "wall": ("concrete", 0.15, config.RT_SCATTERING_COEFFICIENT),     # 0.3 m central walls
    "pillar": ("concrete", 0.30, config.RT_SCATTERING_COEFFICIENT),   # solid concrete pillars
    "cabinet": ("wood", 0.30, config.RT_SCATTERING_COEFFICIENT),      # 0.6 m wooden cabinets
}


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.asarray(vertices, dtype="<f4").tobytes())
        for face in faces:
            handle.write(struct.pack("<B3i", 3, *map(int, face)))


def box_mesh(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box with outward normals."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]])
    return v, f


def quad_mesh(corners) -> tuple[np.ndarray, np.ndarray]:
    """Planar rectangle given by four corners in order."""
    return np.asarray(corners, dtype=float), np.array([[0, 1, 2], [0, 2, 3]])


def cylinder_mesh(cx: float, cy: float, r: float, h: float, n_seg: int = 32) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    ring = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    v = np.vstack([np.c_[ring, np.zeros(n_seg)], np.c_[ring, np.full(n_seg, h)], [[cx, cy, 0.0], [cx, cy, h]]])
    cb, ct = 2 * n_seg, 2 * n_seg + 1
    faces = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces += [[i, j, j + n_seg], [i, j + n_seg, i + n_seg], [cb, j, i], [ct, i + n_seg, j + n_seg]]
    return v, np.array(faces)


def room_surfaces() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Inner faces of the room lie exactly on x=0, x=18, y=0, y=10, z=0 and z=3."""
    X, Y, Z = config.ROOM_X_M, config.ROOM_Y_M, config.ROOM_Z_M
    return {
        "wall_x0": quad_mesh([(0, 0, 0), (0, Y, 0), (0, Y, Z), (0, 0, Z)]),
        "wall_x18": quad_mesh([(X, 0, 0), (X, 0, Z), (X, Y, Z), (X, Y, 0)]),
        "wall_y0": quad_mesh([(0, 0, 0), (0, 0, Z), (X, 0, Z), (X, 0, 0)]),
        "wall_y10": quad_mesh([(0, Y, 0), (X, Y, 0), (X, Y, Z), (0, Y, Z)]),
        "floor": quad_mesh([(0, 0, 0), (X, 0, 0), (X, Y, 0), (0, Y, 0)]),
        "ceiling": quad_mesh([(0, 0, Z), (0, Y, Z), (X, Y, Z), (X, 0, Z)]),
    }


def obstacle_class(obstacle: config.Obstacle) -> str:
    if obstacle.name.startswith("Wall"):
        return "wall"
    return "pillar" if obstacle.kind == "cylinder" else "cabinet"


def obstacle_mesh(obstacle: config.Obstacle) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = obstacle.center_xy_m
    if obstacle.kind == "box":
        assert obstacle.size_xyz_m is not None
        dx, dy, dz = obstacle.size_xyz_m
        return box_mesh((cx - dx / 2, cy - dy / 2, 0.0), (cx + dx / 2, cy + dy / 2, dz))
    assert obstacle.radius_m is not None
    return cylinder_mesh(cx, cy, obstacle.radius_m, obstacle.height_m)


def _materials_xml() -> str:
    out = ""
    for material_id, (itu_type, thickness, scattering) in SURFACE_MATERIALS.items():
        out += (f'  <bsdf type="itu-radio-material" id="mat-{material_id}">\n'
                f'    <string name="type" value="{itu_type}"/>\n'
                f'    <float name="thickness" value="{thickness}"/>\n'
                f'    <float name="scattering_coefficient" value="{scattering}"/>\n'
                f'  </bsdf>\n')
    return out


def _shape_xml(name: str, material_id: str) -> str:
    return (f'  <shape type="ply" id="{name}">\n'
            f'    <string name="filename" value="meshes/{name}.ply"/>\n'
            f'    <ref id="mat-{material_id}" name="bsdf"/>\n'
            f'  </shape>\n')


def build_scene_files(scene_dir: Path = config.SCENE_DIR) -> tuple[Path, Path]:
    """Write meshes and the two Mitsuba XML files; return (rt_xml, render_xml)."""
    mesh_dir = scene_dir / "meshes"
    shapes, render_shapes = "", ""
    for name, mesh in room_surfaces().items():
        _write_ply(mesh_dir / f"{name}.ply", *mesh)
        shapes += _shape_xml(name, "room")
        if name != "ceiling":
            render_shapes += _shape_xml(name, "room")
    for obstacle in config.OBSTACLES:
        name = "obs_" + obstacle.name.replace(" ", "_").lower()
        _write_ply(mesh_dir / f"{name}.ply", *obstacle_mesh(obstacle))
        shapes += _shape_xml(name, obstacle_class(obstacle))
        render_shapes += _shape_xml(name, obstacle_class(obstacle))
    rt_xml, render_xml = scene_dir / "room_rt.xml", scene_dir / "room_render.xml"
    rt_xml.write_text('<scene version="3.0.0">\n' + _materials_xml() + shapes + "</scene>\n", encoding="utf-8")
    render_xml.write_text('<scene version="3.0.0">\n' + _materials_xml() + render_shapes + "</scene>\n", encoding="utf-8")
    return rt_xml, render_xml


if __name__ == "__main__":
    print(build_scene_files())
