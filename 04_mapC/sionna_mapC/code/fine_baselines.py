"""두 기준선을 10 cm 맵으로 14만 조합 전부 다시 계산하고 04/05번 그림의 10 cm 버전을 만든다.

* 4대 동시 출발 그대로 (사용자 결정 2026-11-03).
* R_min이 같은 조합이 많아서(출발 직후 공통 병목), 동률이면 '순간 최저 rate의 시간 평균'이 큰 조합을 고른다.
출력: figures/04b_mumimo_best_10cm.png, 05b_ofdma_best_10cm.png, 08_baselines_10cm_summary.json,
      code/fine_baselines_results.npy
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import mumimo
import ofdma
from collision import AMRS
from fine_eval import fine_sequences, load_fine, placements
from fine_map import FINE_RES_M
from ofdma import FIXED, band_rate_table, best_assignment
from path_planner import load_path_sets

CHUNK = 5_000


def main():
    t0 = time.time()
    meta, H, h_flat, remap, ny, nx = load_fine()
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = fine_sequences(paths, t, nx, ny, remap)
    table = band_rate_table(h_flat)

    n = len(combos)
    o_rmin, o_mean, o_fixed = np.empty(n), np.empty(n), np.empty(n)
    m_rmin, m_mean = np.empty(n), np.empty(n)
    for a in range(0, n, CHUNK):
        c4, inv = placements(seq, combos[a:a + CHUNK])
        va, _ = best_assignment(table, c4)
        o = va[inv]
        o_rmin[a:a + CHUNK], o_mean[a:a + CHUNK] = o.min(axis=1), o.mean(axis=1)
        o_fixed[a:a + CHUNK] = table[c4, FIXED[None, :]].min(axis=1)[inv].min(axis=1)
        m = mumimo.zf_rates_bps(h_flat, c4).min(axis=1)[inv]
        m_rmin[a:a + CHUNK], m_mean[a:a + CHUNK] = m.min(axis=1), m.mean(axis=1)
        done = min(a + CHUNK, n)
        el = time.time() - t0
        print(f"{done:,}/{n:,} combos  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)

    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0)
    total = sum(lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS))
    i_ref = int(np.argmin(makespan * 1000 + total))
    i_m = int(np.lexsort((m_mean, m_rmin))[-1])
    i_o = int(np.lexsort((o_mean, o_rmin))[-1])
    tie = lambda r: int(np.isclose(r, r.max(), rtol=0, atol=1e3).sum())

    # ---- 그림: 기존 plot 함수에 10 cm 격자 정보만 넘긴다 ----
    lookup = dict(zip(map(tuple, combos.tolist()), m_rmin))
    res_m = {"t": t, "seq": seq, "combos": combos, "paths": paths, "h_flat": h_flat,
             "rmin_of": lambda c: lookup[tuple(int(v) for v in c)]}
    mumimo.plot(res_m, combos[i_m], combos[i_ref], config.FIGURES_DIR / "04b_mumimo_best_10cm.png",
                nx=nx, res_m=FINE_RES_M, tag="10 cm map")
    res_o = {"t": t, "seq": seq, "combos": combos, "paths": paths, "table": table,
             "rmin_adapt": o_rmin, "rmin_fixed": o_fixed}
    ofdma.plot(res_o, combos[i_o], config.FIGURES_DIR / "05b_ofdma_best_10cm.png",
               nx=nx, res_m=FINE_RES_M, tag="10 cm map")

    tr_m = mumimo.trace_combo(res_m, combos[i_m])
    tr_o = ofdma.trace_combo(res_o, combos[i_o])
    summary = {
        "channel": "10 cm Sionna map, simultaneous departure",
        "mumimo": {"best_paths": dict(zip(AMRS, map(int, combos[i_m]))),
                   "rmin_mbps": round(float(m_rmin[i_m]) / 1e6, 2),
                   "combos_tied_at_best": tie(m_rmin),
                   "mean_of_instant_min_mbps": round(float(m_mean[i_m]) / 1e6, 2),
                   "mean_rate_per_amr_mbps": dict(zip(AMRS, np.round(tr_m["zf"].mean(axis=0) / 1e6, 2).tolist())),
                   "makespan_s": round(float(makespan[i_m]) / config.AMR_SPEED_MPS, 1),
                   "fastest_combo_rmin_mbps": round(float(m_rmin[i_ref]) / 1e6, 2),
                   "median_over_combos_mbps": round(float(np.median(m_rmin)) / 1e6, 2)},
        "ofdma_adaptive": {"best_paths": dict(zip(AMRS, map(int, combos[i_o]))),
                           "rmin_mbps": round(float(o_rmin[i_o]) / 1e6, 2),
                           "combos_tied_at_best": tie(o_rmin),
                           "mean_of_instant_min_mbps": round(float(o_mean[i_o]) / 1e6, 2),
                           "band_switch_count": int((np.diff(tr_o["bands"], axis=0) != 0).any(axis=1).sum()),
                           "median_over_combos_mbps": round(float(np.median(o_rmin)) / 1e6, 2)},
        "ofdma_fixed": {"best_rmin_mbps": round(float(o_fixed.max()) / 1e6, 2),
                        "median_over_combos_mbps": round(float(np.median(o_fixed)) / 1e6, 2)},
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / "fine_baselines_results.npy",
            {"combos": combos, "mumimo_rmin": m_rmin, "mumimo_mean": m_mean, "ofdma_rmin": o_rmin,
             "ofdma_mean": o_mean, "ofdma_fixed_rmin": o_fixed}, allow_pickle=True)
    (config.FIGURES_DIR / "08_baselines_10cm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
