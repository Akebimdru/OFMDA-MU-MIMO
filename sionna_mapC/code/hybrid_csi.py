"""제안 방식 하이브리드 2×2 (이전 맵과 같은 알고리즘, hybrid._bisect_k): 완벽한 CSI와 CSI 오차 NMSE −40 dB.

알고리즘 (2026-11-08 합의, 그대로 사용)
* 0.1 s마다 쌍 6개의 부반송파별 2-user ZF(BS 4안테나) rate 누적합 → 쌍 구성 3 × 대역 순서 2 = 옵션 6가지,
  옵션마다 k를 이분 탐색 → 순간 R_min 최대 옵션 선택. 조합 성능 = 순간 R_min의 최솟값 (같으면 시간 평균).

CSI 오차 모델 (mumimo_csi.py, ofdma_var_csi.py와 같은 모델)
* 대역 안의 두 AMR a, b에 대해 [(HᴴH)⁻¹]_aa = 1 / (g_aa − |g_ab|²/g_bb),  g = HᴴH.
* 추정 오차 원소 분산 σ_e,j² = NMSE · g_jj / N_ant. ZF 뒤 남는 간섭 평균 전력을 잡음처럼 취급:
      SINR_a = P (g_aa − |g_ab|²/g_bb) / (σ² + P (σ_e,a² + σ_e,b²))
  NMSE = 0 이면 이전 hybrid.pair_prefix와 같다. 간섭은 같은 대역의 상대 1대(와 자기 오차)에서만 온다.
* BS는 NMSE를 알고 위 SINR로 쌍과 k를 정한다고 가정한다.

출력 (완벽한 CSI와 CSI 오차를 따로 저장)
* code/hybrid_perfect_results.npy, figures/11_hybrid_perfect.png, figures/11_hybrid_perfect_summary.json
* code/hybrid_csi40_results.npy,   figures/12_hybrid_csi40.png,   figures/12_hybrid_csi40_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import mumimo_csi
from collision import AMRS
from fine_eval import placements
from hybrid import OPTION_LABEL, PAIRS, _bisect_k, option_rates, plot
from mumimo import SNR_SCALE
from ofdma_var import setup

NMSE_DB = -40.0


def pair_prefix_csi(h_flat, cells4, nmse_lin):
    """(u, 4 AMR, 4 상대 AMR, 65): AMR i가 j와 짝일 때 부반송파 rate 누적합 [bps]."""
    h = h_flat[cells4]                                                   # (u, 4 user, ant, 64)
    g = np.einsum("uian,ujan->uijn", np.conj(h), h)
    gd = np.real(np.einsum("uiin->uin", g))                              # (u, 4, 64)
    pre = np.zeros((len(cells4), 4, 4, config.N_SC + 1))
    for i, j in PAIRS:
        c2 = np.abs(g[:, i, j]) ** 2
        err = 1.0 + SNR_SCALE * nmse_lin * (gd[:, i] + gd[:, j]) / config.N_ANT_BS
        for a, b in ((i, j), (j, i)):
            sinr = SNR_SCALE * np.maximum(gd[:, a] - c2 / np.maximum(gd[:, b], 1e-300), 0.0) / err
            pre[:, a, b, 1:] = np.cumsum(config.DELTA_F_HZ * np.log2(1.0 + sinr), axis=1)
    return pre


def best_hybrid_csi(h_flat, cells4, nmse_lin, chunk=50_000, verbose=False):
    n = len(cells4)
    val, opt, kk = np.empty(n), np.empty(n, np.int8), np.empty(n, np.int8)
    t0 = time.time()
    for a in range(0, n, chunk):
        v, k = _bisect_k(pair_prefix_csi(h_flat, cells4[a:a + chunk], nmse_lin))
        o = v.argmax(axis=1)
        rows = np.arange(len(v))
        val[a:a + chunk], opt[a:a + chunk], kk[a:a + chunk] = v[rows, o], o, k[rows, o]
        if verbose and (a // chunk) % 20 == 0:
            done = min(a + chunk, n); el = time.time() - t0
            print(f"{done:,}/{n:,}  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)
    return val, opt, kk


def evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, nmse_db, name, fig_name, title_tag, other):
    t0 = time.time()
    lin = 0.0 if nmse_db is None else 10 ** (nmse_db / 10)
    val, _, _ = best_hybrid_csi(h_flat, c4, lin, verbose=True)
    inst = val[inv]
    rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
    del inst
    i_h = int(np.lexsort((rmean, rmin))[-1])
    best = combos[i_h]

    cells = np.stack([seq[amr][best[k]] for k, amr in enumerate(AMRS)], axis=1)
    _, o, k = best_hybrid_csi(h_flat, cells, lin)
    rates = option_rates(pair_prefix_csi(h_flat, cells, lin))[np.arange(len(cells)), o, k - 1]
    gd, pw = mumimo_csi.zf_terms(h_flat, cells)
    mm = mumimo_csi.rates_bps(gd, pw, np.array([lin]))[0].min(axis=1)
    tr = {"cells": cells, "rates": rates, "opt": o, "k": k, "mumimo_min": mm}

    rmin_m, rmin_o, rmin_o16 = other
    plot(t, paths, nx, best, tr, rmin, rmin_m, rmin_o, config.FIGURES_DIR / f"{fig_name}.png", tag=title_tag)

    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, j]] for j, amr in enumerate(AMRS)]), axis=0) / config.AMR_SPEED_MPS
    mb = lambda v: round(float(v) / 1e6, 2)
    tw = int(np.argmin(rates.min(axis=1)))
    summary = {
        "map_folder": config.PROJECT_ROOT.name,
        "csi": "perfect" if nmse_db is None else f"NMSE {nmse_db:g} dB",
        "csi_model": "SINR_a = P (g_aa - |g_ab|^2/g_bb) / (s2 + P*NMSE*(g_aa+g_bb)/N_ant)",
        "algorithm": "every 0.1 s: per-SC 2-user ZF for 6 pairs, 3 pairings x 2 band orders, k by bisection, "
                     "maximize instantaneous min rate",
        "feasible_combos": int(len(combos)),
        "hybrid": {"best_paths": dict(zip(AMRS, map(int, best))), "rmin_mbps": mb(rmin[i_h]),
                   "combos_tied_at_best": int(np.isclose(rmin, rmin.max(), rtol=0, atol=1e3).sum()),
                   "mean_of_instant_min_mbps": mb(rmean[i_h]), "makespan_s": round(float(makespan[i_h]), 1),
                   "bottleneck_time_s": round(float(t[tw]), 1), "bottleneck_amr": AMRS[int(rates[tw].argmin())],
                   "k_range": [int(k.min()), int(k.max())],
                   "option_share_pct": dict(zip(OPTION_LABEL, np.round(np.bincount(o, minlength=6) / len(o) * 100, 1).tolist())),
                   "mumimo_on_same_paths_rmin_mbps": mb(rmin_m[i_h]),
                   "median_over_combos_mbps": mb(np.median(rmin))},
        "comparison_best_rmin_mbps": {"ofdma_fixed16": mb(rmin_o16.max()), "ofdma_variable": mb(rmin_o.max()),
                                      "hybrid_2x2": mb(rmin.max()), "mumimo_4x4": mb(rmin_m.max())},
        "comparison_median_over_combos_mbps": {"ofdma_variable": mb(np.median(rmin_o)), "hybrid_2x2": mb(np.median(rmin)),
                                               "mumimo_4x4": mb(np.median(rmin_m))},
        "hybrid_beats_mumimo_pct_of_combos": round(float(np.mean(rmin > rmin_m) * 100), 2),
        "hybrid_beats_ofdma_pct_of_combos": round(float(np.mean(rmin > rmin_o) * 100), 2),
        "unique_placements": int(len(c4)),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / f"{name}.npy",
            {"combos": combos, "rmin": rmin, "rmean": rmean, "nmse_db": nmse_db}, allow_pickle=True)
    (config.FIGURES_DIR / f"{fig_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1), flush=True)


def main():
    h_flat, paths, combos, t, seq, nx = setup()
    c4, inv = placements(seq, combos)
    print(f"{len(c4):,} unique placements", flush=True)
    code = config.PROJECT_ROOT / "code"
    ld = lambda f: np.load(code / f, allow_pickle=True).item()
    mp, mc = ld("mumimo_perfect_results.npy"), ld("mumimo_csi_results.npy")
    op, oc = ld("ofdma_var_perfect_results.npy"), ld("ofdma_var_csi40_results.npy")
    for d in (mp, mc, op, oc):
        assert np.array_equal(d["combos"], combos)
    i40 = int(np.flatnonzero(mc["nmse_db"] == NMSE_DB)[0])
    evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, None, "hybrid_perfect_results", "11_hybrid_perfect",
             "10 cm map, perfect CSI", (mp["rmin"], op["rmin"], op["fixed16_rmin"]))
    evaluate(h_flat, paths, combos, t, seq, nx, c4, inv, NMSE_DB, "hybrid_csi40_results", "12_hybrid_csi40",
             f"10 cm map, CSI NMSE {NMSE_DB:g} dB", (mc["rmin"][i40], oc["rmin"], oc["fixed16_rmin"]))


if __name__ == "__main__":
    main()
