"""OFDMA + 가변 대역 크기 (이전 맵과 같은 알고리즘, ofdma_var.balance): 완벽한 CSI와 CSI 오차 NMSE −40 dB.

알고리즘 (사용자 설계, 2026-11-07, 그대로 사용)
* 0.1 s마다 대역 순서 24가지 각각: 16개씩 4등분에서 시작 → rate 최대 AMR에서 최소 AMR로 부반송파 1개씩
  연속 경계를 밀어 옮기고, 순간 R_min이 엄격히 커질 때만 받아들인다 → 24가지 중 순간 R_min 최대.
* 조합 성능 = 이동 동안 순간 R_min의 최솟값, 같으면 시간 평균이 큰 조합.

CSI 오차 모델 (mumimo_csi.py와 같은 모델, 사용자가 1명인 경우)
* AMR마다 자기 대역을 혼자 쓰므로 AMR 간 간섭은 없다. BS는 추정 채널 ĥ = h − e로 MRC를 한다.
* e의 원소 분산 σ_e² = NMSE · ‖h[n]‖² / N_ant. 추정 오차 성분을 잡음처럼 취급하면
      SINR[n] = P‖h[n]‖² / (σ² + P σ_e²)
  (4x4 ZF 식에서 사용자가 1명이면 [(hᴴh)⁻¹] = 1/‖h‖²가 되어 같은 식이 된다.)
* BS는 오차의 통계(NMSE)를 알고 위 SINR로 k를 정한다고 가정한다.

출력 (완벽한 CSI와 CSI 오차를 따로 저장)
* code/ofdma_var_perfect_results.npy, figures/09_ofdma_var_perfect.png, figures/09_ofdma_var_perfect_summary.json
* code/ofdma_var_csi40_results.npy,   figures/10_ofdma_var_csi40.png,   figures/10_ofdma_var_csi40_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
from collision import AMRS
from fine_eval import placements
from fine_map import FINE_RES_M
from mumimo import SNR_SCALE
from ofdma_var import plot, run_all, setup

NMSE_DB = -40.0


def sc_prefix_table_csi(h_flat: np.ndarray, nmse_lin: float) -> np.ndarray:
    """(점, 65) 부반송파별 MRC rate 누적합 [bps]. nmse_lin = 0이면 완벽한 CSI (ofdma_var.sc_prefix_table과 같음)."""
    g = np.sum(np.abs(h_flat) ** 2, axis=1)                                   # (점, 64) ‖h[n]‖²
    sinr = SNR_SCALE * g / (1.0 + SNR_SCALE * nmse_lin * g / config.N_ANT_BS)
    se = np.log2(1.0 + sinr)
    return np.concatenate([np.zeros((len(se), 1)), np.cumsum(config.DELTA_F_HZ * se, axis=1)], axis=1)


def evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, nmse_db, name, fig_name, title_tag):
    t0 = time.time()
    lin = 0.0 if nmse_db is None else 10 ** (nmse_db / 10)
    prefix = sc_prefix_table_csi(h_flat, lin)
    val, order, sizes, fval, iters = run_all(prefix, c4)
    inst, finst = val[inv], fval[inv]
    rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
    frmin, frmean = finst.min(axis=1), finst.mean(axis=1)
    del inst, finst
    i_v = int(np.lexsort((rmean, rmin))[-1])
    i_f = int(np.lexsort((frmean, frmin))[-1])
    tie = lambda r: int(np.isclose(r, r.max(), rtol=0, atol=1e3).sum())
    best = combos[i_v]
    tr = plot(t, paths, seq, prefix, nx, best, rmin, frmin, config.FIGURES_DIR / f"{fig_name}.png", FINE_RES_M, tag=title_tag)
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0) / config.AMR_SPEED_MPS
    mb = lambda v: round(float(v) / 1e6, 2)
    summary = {
        "map_folder": config.PROJECT_ROOT.name,
        "csi": "perfect" if nmse_db is None else f"NMSE {nmse_db:g} dB",
        "csi_model": "SINR[n] = P|h[n]|^2 / (s2 + P*NMSE*|h[n]|^2/N_ant)",
        "algorithm": "per 0.1 s and per band order (24): start 16 each, shift contiguous boundaries one SC from max-rate "
                     "to min-rate AMR while instantaneous R_min strictly increases; best order",
        "feasible_combos": int(len(combos)),
        "variable_size": {"best_paths": dict(zip(AMRS, map(int, best))), "rmin_mbps": mb(rmin[i_v]),
                          "combos_tied_at_best": tie(rmin), "mean_of_instant_min_mbps": mb(rmean[i_v]),
                          "makespan_s": round(float(makespan[i_v]), 1),
                          "same_paths_16each_rmin_mbps": mb(frmin[i_v]),
                          "band_size_range_per_amr": {amr: [int(tr["sizes"][:, k].min()), int(tr["sizes"][:, k].max())]
                                                      for k, amr in enumerate(AMRS)},
                          "median_over_combos_mbps": mb(np.median(rmin))},
        "fixed16_best_of_24": {"best_paths": dict(zip(AMRS, map(int, combos[i_f]))), "rmin_mbps": mb(frmin[i_f]),
                               "median_over_combos_mbps": mb(np.median(frmin))},
        "unique_placements": int(len(c4)), "max_iterations": int(iters),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / f"{name}.npy",
            {"combos": combos, "rmin": rmin, "rmean": rmean, "fixed16_rmin": frmin, "nmse_db": nmse_db}, allow_pickle=True)
    (config.FIGURES_DIR / f"{fig_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


def main():
    h_flat, paths, combos, t, seq, nx = setup()
    c4, inv = placements(seq, combos)
    print(f"{len(c4):,} unique placements", flush=True)
    evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, None, "ofdma_var_perfect_results",
             "09_ofdma_var_perfect", "10 cm map, perfect CSI")
    evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, NMSE_DB, "ofdma_var_csi40_results",
             "10_ofdma_var_csi40", f"10 cm map, CSI NMSE {NMSE_DB:g} dB")


if __name__ == "__main__":
    main()
