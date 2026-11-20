"""QoS R_min ≥ 120 Mbps + 이동 시간 최소: 이동 시간 짧은 순으로 검사하다 처음 만족하는 곳에서 멈춘다.

* 조합을 (makespan, 총 이동 거리) 오름차순으로 정렬하고, 같은 (makespan, 거리) 묶음 단위로 검사한다.
* 조합 하나 검사 = 4대 모두 도착할 때까지 0.1 s마다 순간 최저 rate 계산, 한 번이라도 120 미만이면 탈락.
* 만족하는 조합이 나온 첫 묶음에서 멈추고, 그 안에서 R_min 최대를 고른다
  → qos_min_time.py(전부 검사)와 같은 규칙이라 같은 답이 나와야 한다.
출력: figures/14_qos120_sorted_{mumimo,hybrid}.png, figures/14_qos120_sorted_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import final_figures as ff
from collision import AMRS
from ofdma_var import setup
from qos_min_time import METHODS, QOS_BPS


def check_group(h, combos, seq, steps, idx, inst_fun):
    """idx 조합들을 끝까지 평가: (통과 여부, R_min)."""
    T = int(steps[idx].max()) + 1
    cells = np.stack([seq[amr][combos[idx, k], :T] for k, amr in enumerate(AMRS)], axis=-1)   # (m, T, 4)
    valid = np.arange(T)[None, :] <= steps[idx][:, None]
    flat = cells[valid]
    code = sum(flat[:, k].astype(np.int64) << (14 * (3 - k)) for k in range(4))
    uniq, first, inv = np.unique(code, return_index=True, return_inverse=True)
    v = np.full(valid.shape, np.inf)
    v[valid] = inst_fun(h, flat[first])[inv.ravel()]
    rmin = v.min(axis=1)
    return rmin >= QOS_BPS, rmin, len(uniq)


def main():
    h_flat, paths, combos, t, seq, nx = setup()
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    summary = {"qos_rmin_mbps": QOS_BPS / 1e6, "path_combinations": int(len(combos)),
               "rule": "sort by (makespan, total length); check groups in order; stop at first group with a QoS-feasible combo; max R_min inside it"}
    for name, label, inst_fun, tfun in METHODS:
        t0 = time.time()
        L = np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)], axis=1)
        makespan = L.max(axis=1) / config.AMR_SPEED_MPS
        total = L.sum(axis=1)
        steps = np.ceil(makespan / config.SIM_DT_S - 1e-9).astype(int)
        order = np.lexsort((total, makespan))
        key = np.round(makespan[order], 6) * 1e4 + np.round(total[order], 6)
        bounds = np.flatnonzero(np.r_[True, np.diff(key) != 0, True])
        checked = groups = n_eval = 0
        i_b = None
        for g in range(len(bounds) - 1):
            idx = order[bounds[g]:bounds[g + 1]]
            ok, rmin, ne = check_group(h_flat, combos, seq, steps, idx, inst_fun)
            checked += len(idx); groups += 1; n_eval += ne
            if ok.any():
                j = np.flatnonzero(ok)
                i_b = int(idx[j[np.argmax(rmin[j])]])
                break
        calc = time.time() - t0
        best = combos[i_b]
        T = steps[i_b] + 1
        cells = np.stack([seq[amr][best[k]][:T] for k, amr in enumerate(AMRS)], axis=1)
        ff.tr_cells = cells
        rates, extra = tfun({"h": h_flat}, cells)
        title = (f"QoS R_min >= {QOS_BPS / 1e6:.0f} Mbps, minimum travel time (sorted search, stop at first hit) — {label} "
                 f"({checked:,} combos checked)")
        r, tw, mk, travel, mean_min = ff.plot(name, title, t[:T], paths, nx, best, rates, extra, calc, lengths,
                                              checked, config.FIGURES_DIR / f"14_qos120_sorted_{name}.png")
        summary[name] = {"computation_time_s": round(calc, 2),
                         "combos_checked": int(checked), "groups_checked": int(groups),
                         "placements_evaluated": int(n_eval),
                         "best_paths": dict(zip(AMRS, map(int, best))),
                         "makespan_s": round(float(makespan[i_b]), 1),
                         "travel_time_s": dict(zip(AMRS, [round(x, 1) for x in travel])),
                         "total_length_m": round(float(total[i_b]), 2),
                         "rmin_mbps": round(float(r) / 1e6, 2),
                         "bottleneck_time_s": round(float(t[tw]), 1),
                         "mean_of_instant_min_mbps": round(float(mean_min) / 1e6, 2)}
        print(name, json.dumps(summary[name]), flush=True)
    (config.FIGURES_DIR / "14_qos120_sorted_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
