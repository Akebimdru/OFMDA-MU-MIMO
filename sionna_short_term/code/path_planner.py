"""Task 2.1 A* 최단 경로 + Task 2.2 무작위 경유점 기반 후보 경로 30개."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

import config

# 재현성 요구사항(handout p.12, p.39): 후보 경로 생성 전에 seed 42.
np.random.seed(config.RANDOM_SEED)

Cell = tuple[int, int]            # (row, col) = (y index, x index)
SQRT2 = float(np.sqrt(2.0))
MOVES = ((-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0),
         (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2))


@dataclass
class Grid:
    x_m: np.ndarray
    y_m: np.ndarray
    blocked: np.ndarray           # (20, 36) bool

    def cell_of(self, xy) -> Cell:
        return int(np.argmin(np.abs(self.y_m - xy[1]))), int(np.argmin(np.abs(self.x_m - xy[0])))

    def xy_of(self, path: list[Cell]) -> np.ndarray:
        return np.array([[self.x_m[c], self.y_m[r]] for r, c in path])


def _dist_to_obstacle(x: float, y: float, ob: config.Obstacle) -> float:
    if ob.kind == "box":
        dx = max(abs(x - ob.cx) - ob.dx / 2, 0.0)
        dy = max(abs(y - ob.cy) - ob.dy / 2, 0.0)
        return float(np.hypot(dx, dy))
    return max(0.0, float(np.hypot(x - ob.cx, y - ob.cy)) - ob.r)


def build_grid() -> Grid:
    """셀 중심이 장애물 외곽선에서 0.2 m 이내(내부 포함)이면 blocked."""
    xs, ys = config.GRID_X_M, config.GRID_Y_M
    blocked = np.array([[any(_dist_to_obstacle(x, y, ob) <= config.OBSTACLE_BUFFER_M for ob in config.OBSTACLES)
                         for x in xs] for y in ys])
    return Grid(xs, ys, blocked)


def heuristic(a: Cell, b: Cell) -> float:
    """격자 단위 유클리드 거리 (8-연결, 대각 √2 비용에서 admissible)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    if grid.blocked[start] or grid.blocked[goal]:
        return None
    rows, cols = grid.blocked.shape
    counter = 0
    pq = [(heuristic(start, goal), counter, start)]
    came_from: dict[Cell, Cell | None] = {start: None}
    g = {start: 0.0}
    closed: set[Cell] = set()
    while pq:
        _, _, cur = heapq.heappop(pq)
        if cur == goal:                       # pop 시점에 종료 → 최적성 보장
            path = []
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dr, dc, cost in MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid.blocked[nb]:
                continue
            ng = g[cur] + cost
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                came_from[nb] = cur
                counter += 1
                heapq.heappush(pq, (ng + heuristic(nb, goal), counter, nb))
    return None


def path_length_m(path: list[Cell]) -> float:
    steps = np.diff(np.asarray(path), axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum() * config.GRID_RES_M)


def signature(path: list[Cell]) -> tuple:
    return tuple(path[::3]) + (path[-1],)


def candidate_paths(grid: Grid, start: Cell, goal: Cell, k: int = config.N_PATH_CANDIDATES) -> list[list[Cell]]:
    shortest = astar(grid, start, goal)
    if shortest is None:
        raise RuntimeError("A*: no path")
    l_cap = config.MAX_PATH_RATIO * path_length_m(shortest)
    free = np.argwhere(~grid.blocked)
    cands, seen = [shortest], {signature(shortest)}
    for _ in range(30 * k):
        if len(cands) >= k:
            break
        via = tuple(int(v) for v in free[np.random.randint(len(free))])
        a, b = astar(grid, start, via), astar(grid, via, goal)
        if a is None or b is None:
            continue
        path = a + b[1:]
        sig = signature(path)
        if sig in seen or path_length_m(path) > l_cap + 1e-9:
            continue
        seen.add(sig)
        cands.append(path)
    return cands


def build_path_sets() -> tuple[Grid, dict[str, list[list[Cell]]]]:
    """AMR1 → AMR2 순서로 한 난수열에서 후보 30개씩 생성 (seed 42 고정)."""
    np.random.seed(config.RANDOM_SEED)
    grid = build_grid()
    paths = {}
    for amr, (s, g) in config.AMR_START_GOAL.items():
        paths[amr] = candidate_paths(grid, grid.cell_of(s), grid.cell_of(g))
    return grid, paths
