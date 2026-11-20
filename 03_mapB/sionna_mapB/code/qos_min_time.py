"""QoS 조건 R_min ≥ 120 Mbps를 만족하는 경로 조합 중 이동 시간(makespan)이 가장 짧은 것을 고른다.
MU-MIMO(4×4 ZF)와 하이브리드(2×2)에 적용. (사용자 요청 2026-11-08)

가지치기: 0.1 s씩 시간을 진행하면서, 살아 있는 조합들의 순간 최저 rate를 계산하고
120 Mbps 미만이 되는 순간 그 조합을 바로 버린다. 그 시각에 살아 있는 조합들의 4대 위치만
중복 없이 계산한다. 각 조합은 자기 makespan(4대 모두 도착)까지만 평가한다.
선택: 끝까지 살아남은 조합 중 makespan 최소 → 총 이동 거리 최소 → R_min 최대.
출력: figures/13_qos120_{mumimo,hybrid}.png, figures/13_qos120_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import final_figures as ff
import mumimo
from collision import AMRS
from hybrid import best_hybrid
from ofdma_var import setup

QOS_BPS = 120e6


def inst_mumimo(h, cells):
    return mumimo.zf_rates_bps(h, cells).min(axis=1)


def inst_hybrid(h, cells):
    return best_hybrid(h, cells)[0]


METHODS = [("mumimo", "MU-MIMO 4x4 ZF", inst_mumimo, ff.t_mumimo),
           ("hybrid", "Proposed hybrid 2x2", inst_hybrid, ff.t_hybrid)]


def run(h, combos, seq, t, steps_of, inst_fun):
    n = len(combos)
    alive = np.ones(n, bool)
    rmin = np.full(n, np.inf)
    rsum = np.zeros(n)
    alive_curve, n_eval = [], 0
    for ti in range(len(t)):
        idx = np.flatnonzero(alive & (steps_of >= ti))      # 아직 이동 중이고 살아 있는 조합
        if len(idx) == 0:
            alive_curve.append(int(alive.sum()))
            continue
        cells = np.stack([seq[amr][combos[idx, k], ti] for k, amr in enumerate(AMRS)], axis=1)
        code = sum(cells[:, k].astype(np.int64) << (14 * (3 - k)) for k in range(4))
        uniq, first, inv = np.unique(code, return_index=True, return_inverse=True)
        v = inst_fun(h, cells[first])[inv.ravel()]
        n_eval += len(uniq)
        rmin[idx] = np.minimum(rmin[idx], v)
        rsum[idx] += v
        alive[idx[v < QOS_BPS]] = False
        alive_curve.append(int(alive.sum()))
    return alive, rmin, rsum, np.array(alive_curve), n_eval


def main():
    h_flat, paths, combos, t, seq, nx = setup()
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    L = np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)], axis=1)
    makespan = L.max(axis=1) / config.AMR_SPEED_MPS
    total = L.sum(axis=1)
    steps_of = np.ceil(makespan / config.SIM_DT_S - 1e-9).astype(int)       # 마지막 평가 시각 인덱스
    summary = {"qos_rmin_mbps": QOS_BPS / 1e6, "path_combinations": int(len(combos)),
               "rule": "keep combos whose instantaneous min rate stays >= QoS until all 4 arrive; pick min makespan, then min total length, then max R_min",
               "shortest_possible_makespan_s": round(float(makespan.min()), 1)}
    for name, label, inst_fun, tfun in METHODS:
        t0 = time.time()
        alive, rmin, rsum, curve, n_eval = run(h_flat, combos, seq, t, steps_of, inst_fun)
        ok = np.flatnonzero(alive)
        calc = time.time() - t0
        if len(ok) == 0:
            summary[name] = {"feasible_combos": 0, "computation_time_s": round(calc, 1)}
            print(name, "no feasible combo", flush=True)
            continue
        i_b = ok[np.lexsort((-rmin[ok], total[ok], makespan[ok]))[0]]
        best = combos[i_b]
        T = steps_of[i_b] + 1
        cells = np.stack([seq[amr][best[k]][:T] for k, amr in enumerate(AMRS)], axis=1)
        ff.tr_cells = cells
        rates, extra = tfun({"h": h_flat}, cells)
        title = (f"QoS R_min >= {QOS_BPS / 1e6:.0f} Mbps, minimum travel time — {label} "
                 f"({len(ok):,} of {len(combos):,} combos satisfy QoS)")
        r, tw, mk, travel, mean_min = ff.plot(name, title, t[:T], paths, nx, best, rates, extra, calc, lengths,
                                              len(combos), config.FIGURES_DIR / f"13_qos120_{name}.png")
        # 가지치기 없이 같은 평가를 했을 때와 비교할 수 있도록 생존 곡선도 저장
        summary[name] = {"feasible_combos": int(len(ok)),
                         "computation_time_s": round(calc, 1),
                         "placements_evaluated": int(n_eval),
                         "alive_after_s": {f"{s:g}": int(curve[min(int(round(s / config.SIM_DT_S)), len(curve) - 1)])
                                           for s in (0.1, 0.5, 1, 2, 5, 10, 20, 30)},
                         "best_paths": dict(zip(AMRS, map(int, best))),
                         "makespan_s": round(float(makespan[i_b]), 1),
                         "travel_time_s": dict(zip(AMRS, [round(x, 1) for x in travel])),
                         "total_length_m": round(float(total[i_b]), 2),
                         "rmin_mbps": round(float(r) / 1e6, 2),
                         "bottleneck_time_s": round(float(t[tw]), 1),
                         "mean_of_instant_min_mbps": round(float(mean_min) / 1e6, 2),
                         "combos_with_same_min_makespan": int(np.isclose(makespan[ok], makespan[i_b]).sum())}
        print(name, json.dumps(summary[name]), flush=True)
    (config.FIGURES_DIR / "13_qos120_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
