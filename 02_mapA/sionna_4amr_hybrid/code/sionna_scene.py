"""새 과제 맵의 Sionna RT 장면(Mitsuba XML + PLY 메시)을 만든다.

구조는 이전 sionna_map_python_algo/code/sionna_scene.py와 같고, 맵만 바뀌었다.
* scene/room_rt.xml     : 둘레 벽 4면, 바닥, 천장, 장애물 4개 (+ 선택: AMR 원통)
* scene/room_render.xml : 같은 장면에서 천장만 뺀 그림용 버전

Sionna는 메시 면 하나를 slab 하나로 보므로, 닫힌 장애물을 지나는 레이는 면 2개를 지난다.
그래서 면 두께는 장애물 두께의 절반으로 둔다.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import config

# 표면 종류별 (ITU 재질, 면 하나의 slab 두께 [m], 확산 계수)
SURFACE_MATERIALS: dict[str, tuple[str, float, float]] = {
    "room": ("concrete", 0.20, config.RT_SCATTERING_COEFFICIENT),     # 둘레 벽, 바닥, 천장
    "wall": ("concrete", 0.25, config.RT_SCATTERING_COEFFICIENT),     # 두께 0.5 m 콘크리트 벽 2개
    "pillar": ("concrete", 0.40, config.RT_SCATTERING_COEFFICIENT),   # 지름 0.8 m 원기둥 (콘크리트 가정)
    "amr": ("metal", 0.01, 0.0),                                      # AMR 몸체 (금속 가정)
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
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]])
    return v, f


def quad_mesh(corners) -> tuple[np.ndarray, np.ndarray]:
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
    X, Y, Z = config.ROOM_X_M, config.ROOM_Y_M, config.ROOM_Z_M
    return {
        "wall_x0": quad_mesh([(0, 0, 0), (0, Y, 0), (0, Y, Z), (0, 0, Z)]),
        "wall_xmax": quad_mesh([(X, 0, 0), (X, 0, Z), (X, Y, Z), (X, Y, 0)]),
        "wall_y0": quad_mesh([(0, 0, 0), (0, 0, Z), (X, 0, Z), (X, 0, 0)]),
        "wall_ymax": quad_mesh([(0, Y, 0), (X, Y, 0), (X, Y, Z), (0, Y, Z)]),
        "floor": quad_mesh([(0, 0, 0), (X, 0, 0), (X, Y, 0), (0, Y, 0)]),
        "ceiling": quad_mesh([(0, 0, Z), (0, Y, Z), (X, Y, Z), (X, 0, Z)]),
    }


def obstacle_class(obstacle: config.Obstacle) -> str:
    return "wall" if obstacle.kind == "box" else "pillar"


def obstacle_mesh(obstacle: config.Obstacle) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = obstacle.center_xy_m
    if obstacle.kind == "box":
        dx, dy, dz = obstacle.size_xyz_m
        return box_mesh((cx - dx / 2, cy - dy / 2, 0.0), (cx + dx / 2, cy + dy / 2, dz))
    return cylinder_mesh(cx, cy, obstacle.radius_m, obstacle.height_m)


def amr_mesh(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    return cylinder_mesh(x, y, config.AMR_RADIUS_M, config.AMR_HEIGHT_M, n_seg=16)


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


def build_scene_files(scene_dir: Path = config.SCENE_DIR,
                      amr_xy_m: dict[str, tuple[float, float]] | None = None) -> tuple[Path, Path]:
    """메시와 XML 2개를 쓰고 (rt_xml, render_xml)을 돌려준다.

    amr_xy_m을 주면 그 위치에 AMR 원통(반지름 0.25 m, 높이 0.5 m)도 장면에 넣는다.
    """
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
    for amr, (x, y) in (amr_xy_m or {}).items():
        name = "amr_" + amr.lower()
        _write_ply(mesh_dir / f"{name}.ply", *amr_mesh(x, y))
        shapes += _shape_xml(name, "amr")
        render_shapes += _shape_xml(name, "amr")
    rt_xml, render_xml = scene_dir / "room_rt.xml", scene_dir / "room_render.xml"
    rt_xml.write_text('<scene version="3.0.0">\n' + _materials_xml() + shapes + "</scene>\n", encoding="utf-8")
    render_xml.write_text('<scene version="3.0.0">\n' + _materials_xml() + render_shapes + "</scene>\n", encoding="utf-8")
    return rt_xml, render_xml


if __name__ == "__main__":
    print(build_scene_files())
