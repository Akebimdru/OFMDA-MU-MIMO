"""OFDMA 탐욕 배분 (사용자 제안, 2026-11-17): 부반송파를 연속 대역으로 묶지 않고 하나씩 나눠 준다.

매 시점(0.1 s), 그 시점의 4대 위치에서
1) AMR1 → AMR2 → AMR3 → AMR4 순서로, 아직 안 쓰인 부반송파 중 자기 채널 ‖h[n]‖²이 가장 큰 것을 1개씩 가져간다.
2) 남은 60개: 지금 rate R이 가장 작은 AMR이 남은 부반송파 중 자기 rate를 가장 키우는 것(= 자기 ‖h[n]‖² 최대)을
   1개 가져간다. 64개가 모두 배정될 때까지 반복한다.
3) 그 시점의 R_min = 4대 중 최저 rate.  경로 조합 성능 = 이동 동안 순간 R_min의 최솟값 (기존과 같은 정의).

* 부반송파별 rate는 ofdma_var_csi.sc_prefix_table_csi와 같은 식 (MRC, 완벽한 CSI 또는 NMSE −40 dB).
* 비교 대상 (같은 CSI 조건, 같은 경로 조합): 경계 이동(기존 사용 알고리즘), 연속 대역 정확한 최적(ofdma_exact).
출력: figures/14_ofdma_greedy_{perfect|csi40}.png, _summary.json, code/ofdma_greedy_{perfect|csi40}_results.npy
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import config
from collision import AMRS
from fine_eval import placements
from mumimo import SNR_SCALE
from ofdma_var import setup
from ofdma_var_csi import NMSE_DB

CHUNK = 100_000


def sc_rate_table(h_flat, nmse_lin):
    """(점, 64) 부반송파별 rate [bps]."""
    g = np.sum(np.abs(h_flat) ** 2, axis=1)
    sinr = SNR_SCALE * g / (1.0 + SNR_SCALE * nmse_lin * g / config.N_ANT_BS)
    return config.DELTA_F_HZ * np.log2(1.0 + sinr)


def greedy(rate_pt, cells4):
    """배치 (u, 4) → (순간 R_min (u,), AMR별 rate (u,4), 부반송파 주인 (u,64))."""
    r = rate_pt[cells4]                                              # (u, 4, 64)
    u = len(cells4)
    rows = np.arange(u)
    taken = np.zeros((u, config.N_SC), bool)
    owner = np.full((u, config.N_SC), -1, np.int8)
    R = np.zeros((u, 4))

    def take(k):
        v = np.where(taken, -np.inf, r[rows, k])
        n = v.argmax(axis=1)
        taken[rows, n] = True
        owner[rows, n] = k
        R[rows, k] += r[rows, k, n]

    for k in range(4):                                               # 1단계: AMR1→4 한 개씩
        take(np.full(u, k))
    for _ in range(config.N_SC - 4):                                 # 2단계: 최저 R인 AMR이 한 개씩
        take(R.argmin(axis=1))
    return R.min(axis=1), R, owner


def main(level):
    t0 = time.time()
    lin = 0.0 if level == "perfect" else 10 ** (NMSE_DB / 10)
    h_flat, paths, combos, t, seq, nx = setup()
    rate_pt = sc_rate_table(h_flat, lin)
    c4, inv = placements(seq, combos)
    n = len(c4)
    t1 = time.time()
    val = np.empty(n)
    for a in range(0, n, CHUNK):
        val[a:a + CHUNK] = greedy(rate_pt, c4[a:a + CHUNK])[0]
    t_alg = time.time() - t1
    inst = val[inv]
    rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
    del inst

    ex = np.load(config.PROJECT_ROOT / "code" / f"ofdma_exact_{level}_results.npy", allow_pickle=True).item()
    assert np.array_equal(ex["combos"], combos)
    r_ex, r_sh = ex["exact_rmin"], ex["shift_rmin"]
    i_g = int(np.lexsort((rmean, rmin))[-1])
    best = combos[i_g]
    mb = lambda v: round(float(v) / 1e6, 2)
    d_sh, d_ex = (rmin - r_sh) / 1e6, (rmin - r_ex) / 1e6

    cells = np.stack([seq[amr][best[k]] for k, amr in enumerate(AMRS)], axis=1)
    _, R_tr, own_tr = greedy(rate_pt, cells)
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0) / config.AMR_SPEED_MPS
    # 연속 대역이 아닌 정도: AMR별 '덩어리' 수 (같은 AMR이 이어지는 구간 수)
    runs = np.array([[len(np.flatnonzero(np.diff(np.r_[-1, np.flatnonzero(o == k)]) != 1)) for k in range(4)] for o in own_tr])
    summary = {
        "map_folder": config.PROJECT_ROOT.name,
        "csi": "perfect" if level == "perfect" else f"NMSE {NMSE_DB:g} dB",
        "algorithm": "per 0.1 s: AMR1..4 each take its best free SC, then the AMR with the lowest current rate takes its best "
                     "free SC until all 64 are assigned (non-contiguous)",
        "feasible_combos": int(n and len(combos)),
        "greedy": {"best_paths": dict(zip(AMRS, map(int, best))), "rmin_mbps": mb(rmin[i_g]),
                   "mean_of_instant_min_mbps": mb(rmean[i_g]), "makespan_s": round(float(makespan[i_g]), 1),
                   "median_over_combos_mbps": mb(np.median(rmin)),
                   "sc_count_range_per_amr": {amr: [int((own_tr == k).sum(1).min()), int((own_tr == k).sum(1).max())]
                                              for k, amr in enumerate(AMRS)},
                   "pieces_per_amr_median": float(np.median(runs))},
        "best_rmin_mbps": {"greedy_noncontiguous": mb(rmin.max()), "boundary_shift_contiguous": mb(r_sh.max()),
                           "exact_contiguous": mb(r_ex.max())},
        "median_over_combos_mbps": {"greedy_noncontiguous": mb(np.median(rmin)),
                                    "boundary_shift_contiguous": mb(np.median(r_sh)),
                                    "exact_contiguous": mb(np.median(r_ex))},
        "per_combo_greedy_minus_shift_mbps": {"mean": round(float(d_sh.mean()), 3), "min": round(float(d_sh.min()), 3),
                                              "max": round(float(d_sh.max()), 3),
                                              "greedy_better_pct": round(float(np.mean(d_sh > 0.01) * 100), 2),
                                              "greedy_worse_pct": round(float(np.mean(d_sh < -0.01) * 100), 2)},
        "per_combo_greedy_minus_exact_contiguous_mbps": {"mean": round(float(d_ex.mean()), 3), "min": round(float(d_ex.min()), 3),
                                                         "max": round(float(d_ex.max()), 3),
                                                         "greedy_better_pct": round(float(np.mean(d_ex > 0.01) * 100), 2),
                                                         "greedy_worse_pct": round(float(np.mean(d_ex < -0.01) * 100), 2)},
        "unique_placements": int(n),
        "greedy_time_per_placement_us": round(t_alg / n * 1e6, 2),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / f"ofdma_greedy_{level}_results.npy",
            {"combos": combos, "rmin": rmin, "rmean": rmean}, allow_pickle=True)
    (config.FIGURES_DIR / f"14_ofdma_greedy_{level}_summary.json").write_text(json.dumps(summary, indent=2))
    plot(level, t, rmin, r_sh, r_ex, best, R_tr, own_tr, summary)
    print(json.dumps(summary, indent=1), flush=True)


def plot(level, t, rmin, r_sh, r_ex, best, R_tr, own_tr, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.9, 0.9])
    ax = fig.add_subplot(gs[0, 0])
    g, sh, ex = rmin / 1e6, r_sh / 1e6, r_ex / 1e6
    bins = np.linspace(min(g.min(), sh.min()), max(g.max(), ex.max()) + 0.5, 80)
    ax.hist(sh, bins=bins, alpha=0.5, label=f"boundary shifting, contiguous (best {sh.max():.2f})")
    ax.hist(ex, bins=bins, histtype="step", color="k", lw=1.2, label=f"exact contiguous optimum (best {ex.max():.2f})")
    ax.hist(g, bins=bins, alpha=0.5, color="tab:red", label=f"greedy, non-contiguous (best {g.max():.2f})")
    ax.set_xlabel("R_min [Mbps]"); ax.set_ylabel("# path combinations"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"R_min over all {len(g):,} collision-free path combinations")
    ax = fig.add_subplot(gs[0, 1])
    d = g - ex
    ax.hist(d, bins=80, color="tab:purple")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("greedy - exact contiguous [Mbps] (per path combination)"); ax.set_ylabel("# combinations"); ax.grid(alpha=0.3)
    pe = s["per_combo_greedy_minus_exact_contiguous_mbps"]
    ax.set_title(f"greedy better in {pe['greedy_better_pct']}%, worse in {pe['greedy_worse_pct']}% (mean {pe['mean']:+.2f} Mbps)")
    ax = fig.add_subplot(gs[1, :])
    for k, amr in enumerate(AMRS):
        ax.plot(t, R_tr[:, k] / 1e6, color=colors[k], lw=1.1, label=amr)
    ax.plot(t, R_tr.min(axis=1) / 1e6, "k", lw=2, label="min over AMRs")
    ax.set_ylabel("rate [Mbps]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=5)
    ax.set_title(f"greedy, best paths {dict(zip(AMRS, map(int, best)))}: R_min {R_tr.min(axis=1).min() / 1e6:.2f} Mbps")
    ax = fig.add_subplot(gs[2, :], sharex=ax)
    ax.imshow(own_tr.T, aspect="auto", origin="lower", cmap=ListedColormap(colors), interpolation="nearest",
              extent=(t[0], t[-1] + (t[1] - t[0]), -0.5, config.N_SC - 0.5))
    ax.set_xlabel("time [s]"); ax.set_ylabel("subcarrier index")
    ax.set_title("which AMR owns each subcarrier (colour = AMR); bands are NOT contiguous")
    fig.suptitle(f"{config.PROJECT_ROOT.name}: OFDMA greedy subcarrier-by-subcarrier allocation vs contiguous bands "
                 f"({'perfect CSI' if level == 'perfect' else f'CSI NMSE {NMSE_DB:g} dB'})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"14_ofdma_greedy_{level}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
