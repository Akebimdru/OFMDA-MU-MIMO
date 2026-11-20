"""Task 2.1-2.2: 8-connected A* planning and reproducible candidate paths."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np

import config
from fingerprint import grid_coordinates_m

Cell = tuple[int, int]  # (row, column)
MOVES: tuple[tuple[int, int, float], ...] = (
    (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
    (0, -1, 1.0),                            (0, 1, 1.0),
    (1, -1, np.sqrt(2.0)),  (1, 0, 1.0),  (1, 1, np.sqrt(2.0)),
)

AMR_ENDPOINTS_M = {
    "AMR1": ((1.0, 2.0), (17.0, 8.0)),
    "AMR2": ((17.0, 2.0), (1.0, 8.0)),
}


@dataclass(frozen=True)
class Grid:
    x_m: np.ndarray
    y_m: np.ndarray
    blocked: np.ndarray

    def nearest_cell(self, point_m: tuple[float, float]) -> Cell:
        """Deterministically map a physical point to its nearest free grid cell."""
        ix = int(np.argmin(np.abs(self.x_m - point_m[0])))
        iy = int(np.argmin(np.abs(self.y_m - point_m[1])))
        if not self.blocked[iy, ix]:
            return iy, ix
        free = np.argwhere(~self.blocked)
        distance_sq = (self.x_m[free[:, 1]] - point_m[0]) ** 2 + (self.y_m[free[:, 0]] - point_m[1]) ** 2
        row, col = free[int(np.argmin(distance_sq))]
        return int(row), int(col)

    def cell_xy_m(self, cell: Cell) -> tuple[float, float]:
        return float(self.x_m[cell[1]]), float(self.y_m[cell[0]])


def _distance_to_box_footprint_m(x_m: float, y_m: float, obstacle: config.Obstacle) -> float:
    assert obstacle.size_xyz_m is not None
    width, height, _ = obstacle.size_xyz_m
    cx, cy = obstacle.center_xy_m
    dx = max(abs(x_m - cx) - width / 2.0, 0.0)
    dy = max(abs(y_m - cy) - height / 2.0, 0.0)
    return float(np.hypot(dx, dy))


def _distance_to_obstacle_footprint_m(x_m: float, y_m: float, obstacle: config.Obstacle) -> float:
    if obstacle.kind == "box":
        return _distance_to_box_footprint_m(x_m, y_m, obstacle)
    assert obstacle.radius_m is not None
    cx, cy = obstacle.center_xy_m
    return max(0.0, float(np.hypot(x_m - cx, y_m - cy) - obstacle.radius_m))


def build_occupancy_grid() -> Grid:
    """Build the handout's 36 x 20 grid with 0.2 m obstacle clearance."""
    x_m, y_m = grid_coordinates_m()
    blocked = np.zeros((len(y_m), len(x_m)), dtype=bool)
    for row, y in enumerate(y_m):
        for col, x in enumerate(x_m):
            blocked[row, col] = any(
                _distance_to_obstacle_footprint_m(x, y, obstacle) <= config.OBSTACLE_BUFFER_M
                for obstacle in config.OBSTACLES
            )
    return Grid(x_m=x_m, y_m=y_m, blocked=blocked)


def _heuristic(cell: Cell, goal: Cell) -> float:
    return float(np.hypot(cell[0] - goal[0], cell[1] - goal[1]))


def astar(grid: Grid, start: Cell, goal: Cell) -> list[Cell] | None:
    """A* with 8-connected costs and an admissible Euclidean grid-unit heuristic."""
    if grid.blocked[start] or grid.blocked[goal]:
        return None
    frontier: list[tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(frontier, (_heuristic(start, goal), counter, start))
    came_from: dict[Cell, Cell | None] = {start: None}
    cost_so_far: dict[Cell, float] = {start: 0.0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:  # Stop only when popped: this certifies optimality.
            path: list[Cell] = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for d_row, d_col, step_cost in MOVES:
            neighbor = current[0] + d_row, current[1] + d_col
            if not (0 <= neighbor[0] < grid.blocked.shape[0] and 0 <= neighbor[1] < grid.blocked.shape[1]):
                continue
            if grid.blocked[neighbor]:
                continue
            candidate_cost = cost_so_far[current] + step_cost
            if candidate_cost < cost_so_far.get(neighbor, np.inf):
                cost_so_far[neighbor] = candidate_cost
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(frontier, (candidate_cost + _heuristic(neighbor, goal), counter, neighbor))
    return None


def path_length_m(path: list[Cell]) -> float:
    return sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(path[:-1], path[1:])) * config.GRID_RES_M


def path_xy_m(grid: Grid, path: list[Cell]) -> np.ndarray:
    return np.asarray([grid.cell_xy_m(cell) for cell in path], dtype=float)


def _signature(path: list[Cell]) -> tuple[Cell, ...]:
    return tuple(path[::3]) + (path[-1],)


def generate_candidate_paths(grid: Grid, start: Cell, goal: Cell, n_candidates: int = config.N_PATH_CANDIDATES) -> list[list[Cell]]:
    """Generate shortest + nonduplicate one-via-cell paths under the 1.5x cap."""
    shortest = astar(grid, start, goal)
    if shortest is None:
        raise RuntimeError("No A* path exists between the requested endpoints")
    shortest_length = path_length_m(shortest)
    candidates = [shortest]
    seen = {_signature(shortest)}
    free_cells = [tuple(map(int, cell)) for cell in np.argwhere(~grid.blocked)]

    for _ in range(30 * n_candidates):
        if len(candidates) >= n_candidates:
            break
        via = free_cells[int(np.random.randint(len(free_cells)))]
        first, second = astar(grid, start, via), astar(grid, via, goal)
        if first is None or second is None:
            continue
        candidate = first + second[1:]
        if path_length_m(candidate) > config.MAX_PATH_RATIO * shortest_length + 1e-9:
            continue
        signature = _signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(candidate)
    if len(candidates) != n_candidates:
        raise RuntimeError(f"Only generated {len(candidates)}/{n_candidates} candidates within the length cap")
    return candidates


def build_path_sets() -> tuple[Grid, dict[str, list[list[Cell]]]]:
    """Build 30 reproducible candidate paths independently for both AMRs."""
    # Required by the handout: seed once before *any* candidate random sampling.
    # Keeping one stream makes AMR2's candidate set the continuation after AMR1's.
    np.random.seed(config.RANDOM_SEED)
    grid = build_occupancy_grid()
    paths: dict[str, list[list[Cell]]] = {}
    for amr, (start_m, goal_m) in AMR_ENDPOINTS_M.items():
        paths[amr] = generate_candidate_paths(grid, grid.nearest_cell(start_m), grid.nearest_cell(goal_m))
    return grid, paths


def save_path_sets(grid: Grid, paths: dict[str, list[list[Cell]]], path: Path | None = None) -> Path:
    """Persist paths as a dictionary-style npy cache for downstream simulator stages."""
    if path is None:
        path = config.PROJECT_ROOT / "code" / "path_candidates.npy"
    payload = {
        "x_m": grid.x_m, "y_m": grid.y_m, "blocked": grid.blocked,
        "AMR1": np.asarray(paths["AMR1"], dtype=object),
        "AMR2": np.asarray(paths["AMR2"], dtype=object),
    }
    np.save(path, payload, allow_pickle=True)
    return path
