"""로봇끼리 충돌하는 경로 조합 거르기.

* 4대가 t=0에 동시에 출발해 0.5 m/s로 각자 경로를 따라가고, 도착하면 목적지에 멈춰 있다.
* 0.1 s마다 두 로봇 중심 거리 < 2 × 반지름(0.5 m)이면 충돌.
* 충돌은 두 대 사이의 성질이라, 4대 조합은 6쌍이 모두 안전해야 안전하다.
  1) 6쌍 각각 26×26 충돌표를 만든다.
  2) 1↔4, 2↔3 (정면으로 마주 오는 쌍) 표로 먼저 거른다.
  3) 남은 (1,4) 쌍 × (2,3) 쌍 조합을 나머지 4쌍(1-2, 1-3, 2-4, 3-4)으로 거른다.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

import config
from path_planner import load_path_sets

AMRS = ("AMR1", "AMR2", "AMR3", "AMR4")
MIN_SEPARATION_M = 2 * config.AMR_RADIUS_M


def positions_on_time_grid(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """폴리라인을 일정 속도로 따라갈 때 시각 t의 위치 (len(t), 2). 도착 후엔 목적지에 정지."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = np.minimum(t * config.AMR_SPEED_MPS, cum[-1])
    return np.c_[np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])]


def build_trajectories(paths: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t_end = max(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    traj = {amr: np.stack([positions_on_time_grid(np.asarray(p, float), t) for p in paths[amr + "_xy"]])
            for amr in AMRS}                                   # (26, T, 2)
    return t, traj


def pair_tables(traj: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """ok[(a,b)][i,j] = a의 i번 경로와 b의 j번 경로가 충돌 없음. dmin = 최소 중심 거리."""
    ok, dmin = {}, {}
    for a, b in itertools.combinations(AMRS, 2):
        d = np.linalg.norm(traj[a][:, None, :, :] - traj[b][None, :, :, :], axis=-1).min(axis=-1)
        dmin[(a, b)] = d
        ok[(a, b)] = d >= MIN_SEPARATION_M
    return ok, dmin


def feasible_combinations(ok: dict) -> np.ndarray:
    """6쌍 모두 안전한 (i1, i2, i3, i4) 조합 (N, 4)."""
    p14 = np.argwhere(ok[("AMR1", "AMR4")])                    # (i1, i4)
    p23 = np.argwhere(ok[("AMR2", "AMR3")])                    # (i2, i3)
    i1, i4 = p14[:, 0][:, None], p14[:, 1][:, None]
    i2, i3 = p23[:, 0][None, :], p23[:, 1][None, :]
    good = (ok[("AMR1", "AMR2")][i1, i2] & ok[("AMR1", "AMR3")][i1, i3]
            & ok[("AMR2", "AMR4")][i2, i4] & ok[("AMR3", "AMR4")][i3, i4])
    a, b = np.nonzero(good)
    return np.c_[p14[a, 0], p23[b, 0], p23[b, 1], p14[a, 1]].astype(np.int16)


def main():
    paths = load_path_sets()
    n = len(paths["AMR1"])
    t, traj = build_trajectories(paths)
    ok, dmin = pair_tables(traj)
    combos = feasible_combinations(ok)

    summary = {"n_paths_per_amr": n, "total_combinations": n ** 4, "min_separation_m": MIN_SEPARATION_M,
               "pairs": {}, "stage1_pairs_14": int(ok[("AMR1", "AMR4")].sum()),
               "stage1_pairs_23": int(ok[("AMR2", "AMR3")].sum())}
    summary["stage1_combinations"] = summary["stage1_pairs_14"] * summary["stage1_pairs_23"]
    summary["feasible_combinations"] = int(len(combos))
    for (a, b), table in ok.items():
        summary["pairs"][f"{a}-{b}"] = {"safe": int(table.sum()), "of": n * n,
                                        "shortest_pair_safe": bool(table[0, 0]),
                                        "shortest_pair_min_dist_m": round(float(dmin[(a, b)][0, 0]), 3)}
    summary["shortest_all_safe"] = bool(len(combos) and (combos == 0).all(axis=1).any())
    if len(combos):
        lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                                  for p in paths[amr + "_xy"]]) for amr in AMRS}
        total = sum(lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS))
        makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0)
        best = combos[np.argmin(makespan * 1000 + total)]
        summary["best_by_makespan"] = {"paths": dict(zip(AMRS, map(int, best))),
                                       "makespan_s": round(float(makespan.min() / config.AMR_SPEED_MPS), 1)}
        summary["count_by_paths_used"] = {amr: int(len(np.unique(combos[:, k]))) for k, amr in enumerate(AMRS)}

    np.save(config.PROJECT_ROOT / "code" / "feasible_combos.npy",
            {"combos": combos, "ok": {f"{a}-{b}": v for (a, b), v in ok.items()},
             "dmin": {f"{a}-{b}": v for (a, b), v in dmin.items()}}, allow_pickle=True)
    (config.FIGURES_DIR / "03_collision_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
