"""Sionna RT(Mitsuba 3)용 실내 창고 장면 생성.

* ``scene/room_rt.xml``     : ray tracing 용. 반사체는 둘레 벽 4개뿐이다(handout p.8).
  바닥/천장은 반사체가 아니므로 넣지 않는다. 장애물은 반사하지 않고 투과 손실(α·chord)만
  주므로 RT 장면에는 넣지 않고, Sionna가 찾은 경로 위에서 fingerprint.py가 손실을 계산한다.
* ``scene/room_render.xml`` : 그림용. 벽 4개 + 장애물 12개 + 바닥.

메시는 외부 패키지 없이 binary PLY로 직접 쓴다.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

import config

WALL_THICKNESS_M = 0.2


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.asarray(vertices, dtype="<f4").tobytes())
        for f in faces:
            fh.write(struct.pack("<B3i", 3, *map(int, f)))


def box_mesh(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """축 정렬 박스 [lo, hi]의 삼각형 메시 (바깥쪽 법선)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2],      # bottom (-z)
                  [4, 5, 6], [4, 6, 7],      # top (+z)
                  [0, 1, 5], [0, 5, 4],      # -y
                  [2, 3, 7], [2, 7, 6],      # +y
                  [1, 2, 6], [1, 6, 5],      # +x
                  [3, 0, 4], [3, 4, 7]])     # -x
    return v, f


def cylinder_mesh(cx, cy, r, h, n_seg: int = 32) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ring = np.c_[cx + r * np.cos(ang), cy + r * np.sin(ang)]
    v = np.vstack([np.c_[ring, np.zeros(n_seg)], np.c_[ring, np.full(n_seg, h)],
                   [[cx, cy, 0.0], [cx, cy, h]]])
    cb, ct = 2 * n_seg, 2 * n_seg + 1
    faces = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces += [[i, j, j + n_seg], [i, j + n_seg, i + n_seg],   # side
                  [cb, j, i], [ct, i + n_seg, j + n_seg]]          # caps
    return v, np.array(faces)


def perimeter_walls() -> dict[str, tuple[tuple, tuple]]:
    """방 안쪽 면이 정확히 x=0, x=18, y=0, y=10 평면이 되도록 바깥쪽으로 두께를 준다."""
    t, X, Y, Z = WALL_THICKNESS_M, config.ROOM_X_M, config.ROOM_Y_M, config.ROOM_Z_M
    return {
        "wall_x0": ((-t, -t, 0.0), (0.0, Y + t, Z)),
        "wall_x18": ((X, -t, 0.0), (X + t, Y + t, Z)),
        "wall_y0": ((-t, -t, 0.0), (X + t, 0.0, Z)),
        "wall_y10": ((-t, Y, 0.0), (X + t, Y + t, Z)),
    }


def _shape_xml(name: str, mat: str) -> str:
    return (f'  <shape type="ply" id="{name}">\n'
            f'    <string name="filename" value="meshes/{name}.ply"/>\n'
            f'    <ref id="mat-{mat}" name="bsdf"/>\n'
            f'  </shape>\n')


def _materials_xml() -> str:
    out = ""
    for mat in ("concrete", "wood"):
        out += (f'  <bsdf type="itu-radio-material" id="mat-{mat}">\n'
                f'    <string name="type" value="{mat}"/>\n'
                f'    <float name="thickness" value="0.2"/>\n'
                f'  </bsdf>\n')
    return out


def build_scene_files(scene_dir: Path = config.SCENE_DIR) -> tuple[Path, Path]:
    mesh_dir = scene_dir / "meshes"
    rt_shapes, render_shapes = "", ""
    for name, (lo, hi) in perimeter_walls().items():
        _write_ply(mesh_dir / f"{name}.ply", *box_mesh(lo, hi))
        rt_shapes += _shape_xml(name, "concrete")
    render_shapes += rt_shapes
    for ob in config.OBSTACLES:
        name = "obs_" + ob.name.replace(" ", "_").lower()
        if ob.kind == "box":
            v, f = box_mesh((ob.cx - ob.dx / 2, ob.cy - ob.dy / 2, 0.0), (ob.cx + ob.dx / 2, ob.cy + ob.dy / 2, ob.height))
        else:
            v, f = cylinder_mesh(ob.cx, ob.cy, ob.r, ob.height)
        _write_ply(mesh_dir / f"{name}.ply", v, f)
        render_shapes += _shape_xml(name, ob.material)
    _write_ply(mesh_dir / "floor.ply", *box_mesh((-0.2, -0.2, -0.05), (config.ROOM_X_M + 0.2, config.ROOM_Y_M + 0.2, 0.0)))
    render_shapes += _shape_xml("floor", "concrete")

    rt_xml, render_xml = scene_dir / "room_rt.xml", scene_dir / "room_render.xml"
    for path, shapes in ((rt_xml, rt_shapes), (render_xml, render_shapes)):
        path.write_text('<scene version="3.0.0">\n' + _materials_xml() + shapes + "</scene>\n", encoding="utf-8")
    return rt_xml, render_xml


if __name__ == "__main__":
    print(build_scene_files())
