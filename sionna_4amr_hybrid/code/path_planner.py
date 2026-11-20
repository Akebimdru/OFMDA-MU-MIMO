"""AMR 4대의 후보 경로: 8방향 격자 A* 최단 경로 1개 + 우회 경로 25개.

이전 sionna_map_python_algo/code/path_planner.py와 같은 방식이다.
* 격자: 0.5 m 셀 중심(fingerprint와 같은 격자), 8방향 이동 (대각선 비용 √2).
* 막힌 셀: 장애물 바닥면까지 거리 ≤ 로봇 반지름 0.25 m.
* 우회 경로: 빈 셀 하나를 무작위로 골라 출발→경유, 경유→도착을 각각 A*로 잇는다.
  최단 경로 길이의 1.5배 이하, 중복(3칸 간격 서명이 같은 경로)은 버린다.
* 출발/도착점 (0.5, 0.5) 등은 셀 중심이 아니라 셀 4개의 꼭짓점이다.
  그래서 방 안쪽 방향의 가장 가까운 셀로 붙이고, 실제 좌표를 경로 양 끝에 덧붙인다(약 0.35 m 대각 구간).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json

import numpy as np

import config

Cell = tuple[int, int]  # (row, column)
MOVES: tuple[tuple[int, int, float], ...] = (
    (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),                            (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),  (1, 0, 1.0),  (1, 1, np.sqrt(2.0)),
)


@dataclass(frozen=True)
class Grid:
    x_m: np.ndarray
    y_m: np.ndarray
    blocked: np.ndarray

    def nearest_cell(self, point_m: tuple[float, float]) -> Cell:
        """가장 가까운 빈 셀. 거리가 같으면 방 가운데 쪽 셀을 고른다."""
        free = np.argwhere(~self.blocked)
        fx, fy = self.x_m[free[:, 1]], self.y_m[free[:, 0]]
        d = np.round((fx - point_m[0]) ** 2 + (fy - point_m[1]) ** 2, 9)
        to_center = (fx - config.ROOM_X_M / 2) ** 2 + (fy - config.ROOM_Y_M / 2) ** 2
        row, col = free[np.lexsort((to_center, d))[0]]
        return int(row), int(col)

    def cell_xy_m(self, cell: Cell) -> tuple[float, float]:
        return float(self.x_m[cell[1]]), float(self.y_m[cell[0]])


def _distance_to_obstacle_footprint_m(x: float, y: float, ob: config.Obstacle) -> float:
    cx, cy = ob.center_xy_m
    if ob.kind == "box":
        w, h, _ = ob.size_xyz_m
        return float(np.hypot(max(abs(x - cx) - w / 2, 0.0), max(abs(y - cy) - h / 2, 0.0)))
    return max(0.0, float(np.hypot(x - cx, y - cy) - ob.radius_m))


def build_occupancy_grid() -> Grid:
    x_m, y_m = config.grid_coordinates_m()
    blocked = np.zeros((len(y_m), len(x_m)), dtype=bool)
    for r, y in enumerate(y_m):
        for c, x in enumerate(x_m):
            blocked[r, c] = any(_distance_to_obstacle_footprint_m(x, y, ob) <= config.OBSTACLE_BUFFER_M
                                for ob in config.OBSTACLES)
    return Grid(x_m=x_m, y_m=y_m, blocked=blocked)


def _heuristic(cell: Cell, goal: Cell) -> float:
    return float(np.hypot(cell[0] - goal[0], cell[1] - goal[1]))


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    if grid.blocked[start] or grid.blocked[goal]:
        return None
    frontier: list[tuple[float, int, Cell]] = [(_heuristic(start, goal), 0, start)]
    counter = 0
    came_from: dict[Cell, Cell | None] = {start: None}
    cost: dict[Cell, float] = {start: 0.0}
    while frontier:
        _, _, cur = heapq.heappop(frontier)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        for dr, dc, step in MOVES:
            nb = cur[0] + dr, cur[1] + dc
            if not (0 <= nb[0] < grid.blocked.shape[0] and 0 <= nb[1] < grid.blocked.shape[1]) or grid.blocked[nb]:
                continue
            new_cost = cost[cur] + step
            if new_cost < cost.get(nb, np.inf):
                cost[nb] = new_cost
                came_from[nb] = cur
                counter += 1
                heapq.heappush(frontier, (new_cost + _heuristic(nb, goal), counter, nb))
    return None


def cells_length_m(path: list[Cell]) -> float:
    return sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(path[:-1], path[1:])) * config.GRID_RES_M


def _signature(path: list[Cell]) -> tuple[Cell, ...]:
    return tuple(path[::3]) + (path[-1],)


def generate_candidate_paths(grid: Grid, start: Cell, goal: Cell,
                             n_candidates: int = config.N_PATH_CANDIDATES) -> list[list[Cell]]:
    shortest = astar(grid, start, goal)
    if shortest is None:
        raise RuntimeError("No A* path between the endpoints")
    cap = config.MAX_PATH_RATIO * cells_length_m(shortest) + 1e-9
    candidates, seen = [shortest], {_signature(shortest)}
    free_cells = [tuple(map(int, c)) for c in np.argwhere(~grid.blocked)]
    for _ in range(30 * n_candidates):
        if len(candidates) >= n_candidates:
            break
        via = free_cells[int(np.random.randint(len(free_cells)))]
        first, second = astar(grid, start, via), astar(grid, via, goal)
        if first is None or second is None:
            continue
        cand = first + second[1:]
        if cells_length_m(cand) > cap or _signature(cand) in seen:
            continue
        seen.add(_signature(cand))
        candidates.append(cand)
    if len(candidates) != n_candidates:
        raise RuntimeError(f"Only {len(candidates)}/{n_candidates} candidates within the length cap")
    return candidates


def path_xy_m(grid: Grid, path: list[Cell], start_m, goal_m) -> np.ndarray:
    """셀 중심 좌표 + 실제 출발/도착 좌표를 양 끝에 붙인 폴리라인 (N, 2)."""
    pts = [tuple(start_m)] + [grid.cell_xy_m(c) for c in path] + [tuple(goal_m)]
    return np.asarray(pts, dtype=float)


def polyline_length_m(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def build_path_sets():
    """AMR1~4 순서로 하나의 난수열(seed 42)을 이어 쓴다 (재현 가능)."""
    np.random.seed(config.RANDOM_SEED)
    grid = build_occupancy_grid()
    cells, xy = {}, {}
    for amr, (start_m, goal_m) in config.AMR_ENDPOINTS_M.items():
        cells[amr] = generate_candidate_paths(grid, grid.nearest_cell(start_m), grid.nearest_cell(goal_m))
        xy[amr] = [path_xy_m(grid, p, start_m, goal_m) for p in cells[amr]]
    return grid, cells, xy


def save_path_sets(grid: Grid, cells, xy) -> None:
    payload = {"x_m": grid.x_m, "y_m": grid.y_m, "blocked": grid.blocked}
    for amr in cells:
        payload[amr] = np.asarray(cells[amr], dtype=object)
        payload[amr + "_xy"] = np.asarray(xy[amr], dtype=object)
    np.save(config.PATH_CANDIDATES_PATH, payload, allow_pickle=True)


def load_path_sets() -> dict:
    return np.load(config.PATH_CANDIDATES_PATH, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# 그림 + 요약
# ---------------------------------------------------------------------------
def plot_candidates(grid: Grid, xy) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    from fingerprint import load_or_build_fingerprint, snr_grid_db

    fp = load_or_build_fingerprint(verbose=False)
    mrc = snr_grid_db(fp["H"], "mrc").mean(axis=2)
    mrc[fp["occupied"]] = np.nan
    extent = (0, config.ROOM_X_M, 0, config.ROOM_Y_M)
    colors = {"AMR1": "tab:blue", "AMR2": "tab:orange", "AMR3": "tab:green", "AMR4": "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (amr, paths) in zip(axes.ravel(), xy.items()):
        im = ax.imshow(mrc, origin="lower", extent=extent, cmap="Greys_r", alpha=0.55, vmin=25, vmax=65)
        ax.imshow(np.where(grid.blocked, 1.0, np.nan), origin="lower", extent=extent, cmap="Reds", alpha=0.25, vmin=0, vmax=1)
        for ob in config.OBSTACLES:
            cx, cy = ob.center_xy_m
            if ob.kind == "box":
                dx, dy, _ = ob.size_xyz_m
                ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray", ec="k"))
            else:
                ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray", ec="k"))
        for p in paths[1:]:
            ax.plot(p[:, 0], p[:, 1], color=colors[amr], lw=1.0, alpha=0.45)
        ax.plot(paths[0][:, 0], paths[0][:, 1], color="k", lw=2.6)
        ax.plot(paths[0][:, 0], paths[0][:, 1], color=colors[amr], lw=1.6, label=f"shortest {polyline_length_m(paths[0]):.2f} m")
        s, g = paths[0][0], paths[0][-1]
        ax.add_patch(Circle(s, config.AMR_RADIUS_M, fill=False, ec=colors[amr], lw=2))
        ax.plot(*g, "*", color=colors[amr], ms=16, mec="k")
        ax.plot(*config.BS_CENTER_M[:2], "w^", ms=10, mec="k")
        lengths = [polyline_length_m(p) for p in paths]
        ax.set_title(f"{amr}: ({s[0]:g},{s[1]:g}) → ({g[0]:g},{g[1]:g}), {len(paths)} paths, "
                     f"length {min(lengths):.1f}–{max(lengths):.1f} m")
        ax.set_xlim(0, config.ROOM_X_M)
        ax.set_ylim(0, config.ROOM_Y_M)
        ax.set_aspect("equal")
        ax.legend(loc="upper center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label="background: MRC SNR [dB]")
    fig.suptitle("Candidate paths: shortest (bold) + 25 detours (≤ 1.5× shortest), 8-connected 0.5 m grid")
    out = config.FIGURES_DIR / "02_candidate_paths.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


def summarize(xy) -> dict:
    out = {}
    for amr, paths in xy.items():
        lengths = np.array([polyline_length_m(p) for p in paths])
        out[amr] = {"start_goal_m": [list(map(float, paths[0][0])), list(map(float, paths[0][-1]))],
                    "shortest_m": round(float(lengths[0]), 3),
                    "shortest_time_s": round(float(lengths[0] / config.AMR_SPEED_MPS), 1),
                    "detour_m": {"min": round(float(lengths[1:].min()), 3), "median": round(float(np.median(lengths[1:])), 3),
                                 "max": round(float(lengths[1:].max()), 3)},
                    "lengths_m": [round(float(v), 3) for v in lengths]}
    (config.FIGURES_DIR / "02_candidate_paths_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    grid, cells, xy = build_path_sets()
    save_path_sets(grid, cells, xy)
    s = summarize(xy)
    for amr, v in s.items():
        print(amr, v["shortest_m"], v["detour_m"])
    plot_candidates(grid, xy)
