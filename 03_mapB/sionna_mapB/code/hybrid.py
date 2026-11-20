"""제안 방식: 하이브리드 2×2 (대역 2개 × AMR 2대, 대역 안에서 BS 4안테나로 2명 ZF).

알고리즘 (사용자와 합의, 2026-11-08, 적응형 없이 2×2만)
* 0.1 s마다 4대 위치에서 쌍 6개(4C2)의 부반송파별 2-user ZF SINR
      SINR_i[n] = (P/σ²)·(g_ii − |g_ij|²/g_jj) = (P/σ²)·‖h_i‖²·(1 − |ρ_ij|²),  g = HᴴH
  → k와 무관하므로 한 번만 계산하고 부반송파 누적합을 만든다.
* 쌍 구성 3가지 × 어느 쌍이 대역 A(0~k−1)를 쓰는지 2가지 = 옵션 6가지. 옵션마다 k(1~63)는
  이분 탐색으로 찾는다: 대역 A 최저 rate는 k에 대해 증가, 대역 B 최저 rate는 감소하므로 교차점 근처가 최적.
  (2026-11-08 사용자 요청으로 378가지 전수 비교에서 변경, 결과는 전수 비교와 동일)
  순간 R_min(4대 중 최저 rate)이 최대인 경우를 고른다 (그 순간의 정확한 최적).
* 경로 조합 성능 = 이동 시간 동안 순간 R_min의 최솟값 (기준선과 같은 정의, 동률이면 시간 평균).
출력: figures/11_hybrid_10cm.png, 11_hybrid_10cm_summary.json, code/hybrid_results.npy
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
from ofdma_var import setup

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]          # 4C2
PAIRINGS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
# 6가지 (쌍 구성, 대역 A 쌍, 대역 B 쌍)
OPTIONS = [(p, a, b) for p, (x, y) in enumerate(PAIRINGS) for a, b in ((x, y), (y, x))]
OPTION_LABEL = [f"A:{a[0]+1}{a[1]+1} B:{b[0]+1}{b[1]+1}" for _, a, b in OPTIONS]


def pair_prefix(h_flat, cells4):
    """(u, 4 AMR, 4 상대 AMR, 65): AMR i가 j와 짝일 때 부반송파 rate 누적합 [bps]."""
    h = h_flat[cells4]                                                   # (u, 4 user, ant, 64)
    g = np.einsum("uian,ujan->uijn", np.conj(h), h)                      # (u, 4, 4, 64)
    gd = np.real(np.einsum("uiin->uin", g))                              # (u, 4, 64)
    u = len(cells4)
    pre = np.zeros((u, 4, 4, config.N_SC + 1))
    for i, j in PAIRS:
        c2 = np.abs(g[:, i, j]) ** 2
        for a, b in ((i, j), (j, i)):
            sinr = SNR_SCALE * np.maximum(gd[:, a] - c2 / np.maximum(gd[:, b], 1e-300), 0.0)
            pre[:, a, b, 1:] = np.cumsum(config.DELTA_F_HZ * np.log2(1.0 + sinr), axis=1)
    return pre


def option_rates(pre):
    """(u, 6 옵션, 63 k, 4 AMR) AMR별 rate."""
    u = pre.shape[0]
    k = np.arange(1, config.N_SC)
    out = np.empty((u, len(OPTIONS), len(k), 4))
    for o, (_, (a1, a2), (b1, b2)) in enumerate(OPTIONS):
        out[:, o, :, a1] = pre[:, a1, a2, k]
        out[:, o, :, a2] = pre[:, a2, a1, k]
        out[:, o, :, b1] = pre[:, b1, b2, -1:] - pre[:, b1, b2, k]
        out[:, o, :, b2] = pre[:, b2, b1, -1:] - pre[:, b2, b1, k]
    return out


def _option_ends(pre):
    """옵션 6가지의 (대역 A 쌍의 누적합 2개, 대역 B 쌍의 누적합 2개): 각 (u, 6, 65)."""
    A1 = np.stack([pre[:, a[0], a[1]] for _, a, _ in OPTIONS], axis=1)
    A2 = np.stack([pre[:, a[1], a[0]] for _, a, _ in OPTIONS], axis=1)
    B1 = np.stack([pre[:, b[0], b[1]] for _, _, b in OPTIONS], axis=1)
    B2 = np.stack([pre[:, b[1], b[0]] for _, _, b in OPTIONS], axis=1)
    return A1, A2, B1, B2


def _fa_fb(ends, k):
    """k (u, 6) 에서 fA = 대역 A 두 AMR 중 최저 rate (k에 대해 증가), fB = 대역 B 최저 (감소)."""
    A1, A2, B1, B2 = ends
    g = lambda P, i: np.take_along_axis(P, i[..., None], axis=2)[..., 0]
    fa = np.minimum(g(A1, k), g(A2, k))
    fb = np.minimum(B1[..., -1] - g(B1, k), B2[..., -1] - g(B2, k))
    return fa, fb


def _bisect_k(pre):
    """옵션마다 min(fA(k), fB(k))를 최대로 하는 k를 이분 탐색으로 찾는다 (사용자 제안, 2026-11-08).
    fA는 k에 대해 단조 증가, fB는 단조 감소이므로 fA(k) >= fB(k)인 가장 작은 k* 또는 k*-1이 최적이다.
    탐색 6번 + 후보 2개 평가 = 옵션당 약 8번 (전수 63번 대신)."""
    ends = _option_ends(pre)
    u = pre.shape[0]
    lo = np.ones((u, 6), int)
    hi = np.full((u, 6), config.N_SC)                 # 64 = '교차점 없음' 표시
    for _ in range(6):                                 # 구간 [1, 64] → 2^6
        mid = (lo + hi) // 2
        fa, fb = _fa_fb(ends, np.minimum(mid, config.N_SC - 1))
        cross = (fa >= fb) & (mid <= config.N_SC - 1)
        hi = np.where(cross, mid, hi)
        lo = np.where(cross, lo, mid + 1)
    k1 = np.minimum(hi, config.N_SC - 1)               # 교차 없으면 63
    k0 = np.maximum(hi - 1, 1)
    v1 = np.minimum(*_fa_fb(ends, k1))
    v0 = np.minimum(*_fa_fb(ends, k0))
    use1 = v1 >= v0
    return np.where(use1, v1, v0), np.where(use1, k1, k0)


def best_hybrid(h_flat, cells4, chunk=50_000, verbose=False, method="bisect"):
    """배치마다 (순간 R_min 최대값, 옵션 번호, k).
    method="bisect": 옵션 6가지마다 k를 이분 탐색 (기본). method="exhaustive": 378가지 전부 (검증용)."""
    n = len(cells4)
    val, opt, kk = np.empty(n), np.empty(n, np.int8), np.empty(n, np.int8)
    t0 = time.time()
    for a in range(0, n, chunk):
        pre = pair_prefix(h_flat, cells4[a:a + chunk])
        if method == "exhaustive":
            r = option_rates(pre).min(axis=3)                                   # (u, 6, 63)
            flat = r.reshape(len(r), -1)
            i = flat.argmax(axis=1)
            val[a:a + chunk] = flat[np.arange(len(r)), i]
            opt[a:a + chunk], kk[a:a + chunk] = i // 63, i % 63 + 1
        else:
            v, k = _bisect_k(pre)                                               # (u, 6)
            o = v.argmax(axis=1)
            rows = np.arange(len(v))
            val[a:a + chunk], opt[a:a + chunk], kk[a:a + chunk] = v[rows, o], o, k[rows, o]
        if verbose and (a // chunk) % 10 == 0:
            done = min(a + chunk, n); el = time.time() - t0
            print(f"{done:,}/{n:,}  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)
    return val, opt, kk


def plot(t, paths, nx, best, tr, rmin_h, rmin_m, rmin_o, out_path, tag="10 cm map"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    rates = tr["rates"]
    tw = int(np.argmin(rates.min(axis=1)))
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 0.55, 0.55], width_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray"))
        else:
            ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray"))
    for k, amr in enumerate(AMRS):
        xy = np.asarray(paths[amr + "_xy"][best[k]], float)
        ax.plot(xy[:, 0], xy[:, 1], color=colors[k], lw=2, label=f"{amr} path #{best[k]}")
        ax.plot(*xy[-1], "*", color=colors[k], ms=14, mec="k")
        r, c = divmod(int(tr["cells"][tw, k]), nx)
        ax.add_patch(Circle(((c + 0.5) * FINE_RES_M, (r + 0.5) * FINE_RES_M), config.AMR_RADIUS_M,
                            fc=colors[k], ec="k", alpha=0.8))
    ax.plot(*config.BS_CENTER_M[:2], "k^", ms=10)
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title(f"best paths, R_min {rates.min() / 1e6:.1f} Mbps\n(circles = positions at bottleneck t={t[tw]:.1f} s)")

    ax = fig.add_subplot(gs[0, 1])
    h, m, o = rmin_h / 1e6, rmin_m / 1e6, rmin_o / 1e6
    bins = np.linspace(min(o.min(), h.min()), max(m.max(), h.max()) + 1, 100)
    ax.hist(o, bins=bins, alpha=0.55, label=f"B1 OFDMA, variable band size (best {o.max():.1f})")
    ax.hist(h, bins=bins, alpha=0.55, label=f"proposed hybrid 2x2 (best {h.max():.1f})")
    ax.hist(m, bins=bins, alpha=0.55, label=f"B2 MU-MIMO 4x4 ZF (best {m.max():.1f})")
    ax.set_xlabel("R_min [Mbps]"); ax.set_ylabel("# path combinations"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"R_min over all {len(h):,} collision-free path combinations")

    ax = fig.add_subplot(gs[1, :])
    for k, amr in enumerate(AMRS):
        ax.plot(t, rates[:, k] / 1e6, color=colors[k], lw=1.2)
    ax.plot(t, rates.min(axis=1) / 1e6, "k", lw=2.2)
    ax.plot(t, tr["mumimo_min"] / 1e6, color="gray", lw=1.2, ls=":")
    ax.axvline(t[tw], color="k", ls="--", lw=0.8)
    handles = [Line2D([], [], color=colors[k], lw=1.2, label=amr) for k, amr in enumerate(AMRS)]
    handles += [Line2D([], [], color="k", lw=2.2, label="min over AMRs (hybrid)"),
                Line2D([], [], color="gray", ls=":", label="MU-MIMO 4x4 min, same paths"),
                Line2D([], [], color="k", ls="--", lw=0.8, label="bottleneck time")]
    ax.legend(handles=handles, fontsize=8, ncol=7, loc="lower center")
    ax.set_ylabel("rate [Mbps]"); ax.grid(alpha=0.3)
    ax.set_title("rates along the best paths (2 bands x 2 AMRs, 4x2 ZF in each band)")

    ax = fig.add_subplot(gs[2, :], sharex=ax)
    ax.step(t, tr["opt"], where="post", color="k", lw=1.3)
    ax.set_yticks(range(6), OPTION_LABEL, fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("pairing and band order chosen (A = subcarriers 0..k-1, B = k..63)")
    ax = fig.add_subplot(gs[3, :], sharex=ax)
    ax.step(t, tr["k"], where="post", color="tab:purple", lw=1.3)
    ax.axhline(32, color="k", ls=":", lw=0.8)
    ax.set_ylabel("k (size of band A)"); ax.set_xlabel("time [s]"); ax.grid(alpha=0.3)
    ax.set_title("band boundary k")
    fig.suptitle("Proposed: hybrid OFDMA + MU-MIMO (2 bands x 2 AMRs, ZF within band, 6 options x bisection on k every 0.1 s) — " + tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    import mumimo
    t0 = time.time()
    h_flat, paths, combos, t, seq, nx = setup()
    c4, inv = placements(seq, combos)
    print(f"{len(c4):,} unique placements", flush=True)
    val, opt, kk = best_hybrid(h_flat, c4, verbose=True)
    inst = val[inv]
    rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
    del inst
    i_h = int(np.lexsort((rmean, rmin))[-1])
    best = combos[i_h]

    cells = np.stack([seq[amr][best[k]] for k, amr in enumerate(AMRS)], axis=1)
    v, o, k = best_hybrid(h_flat, cells)
    rr = option_rates(pair_prefix(h_flat, cells))
    rates = rr[np.arange(len(cells)), o, k - 1]
    mm = mumimo.zf_rates_bps(h_flat, cells).min(axis=1)
    tr = {"cells": cells, "rates": rates, "opt": o, "k": k, "mumimo_min": mm}

    base = np.load(config.PROJECT_ROOT / "code" / "fine_baselines_results.npy", allow_pickle=True).item()
    var = np.load(config.PROJECT_ROOT / "code" / "ofdma_var_results.npy", allow_pickle=True).item()
    assert np.array_equal(base["combos"], combos) and np.array_equal(var["combos"], combos)
    rmin_m, rmin_o = base["mumimo_rmin"], var["rmin"]
    plot(t, paths, nx, best, tr, rmin, rmin_m, rmin_o, config.FIGURES_DIR / "11_hybrid_10cm.png")

    tie = lambda r: int(np.isclose(r, r.max(), rtol=0, atol=1e3).sum())
    tw = int(np.argmin(rates.min(axis=1)))
    opt_share = np.bincount(o, minlength=6) / len(o) * 100
    summary = {
        "channel": "10 cm Sionna map, simultaneous departure",
        "algorithm": "every 0.1 s: per-SC 2-user ZF for 6 pairs, 3 pairings x 2 band orders, k by bisection (fA increasing, fB decreasing in k), maximize instantaneous min rate",
        "hybrid": {"best_paths": dict(zip(AMRS, map(int, best))),
                   "rmin_mbps": round(float(rmin[i_h]) / 1e6, 2),
                   "combos_tied_at_best": tie(rmin),
                   "mean_of_instant_min_mbps": round(float(rmean[i_h]) / 1e6, 2),
                   "bottleneck_time_s": round(float(t[tw]), 1),
                   "bottleneck_amr": AMRS[int(rates[tw].argmin())],
                   "k_range": [int(k.min()), int(k.max())],
                   "option_share_pct": dict(zip(OPTION_LABEL, np.round(opt_share, 1).tolist())),
                   "mumimo_on_same_paths_rmin_mbps": round(float(rmin_m[i_h]) / 1e6, 2),
                   "median_over_combos_mbps": round(float(np.median(rmin)) / 1e6, 2)},
        "comparison_best_rmin_mbps": {"ofdma_fixed16": round(float(var["fixed16_rmin"].max()) / 1e6, 2),
                                      "ofdma_variable": round(float(rmin_o.max()) / 1e6, 2),
                                      "hybrid_2x2": round(float(rmin.max()) / 1e6, 2),
                                      "mumimo_4x4": round(float(rmin_m.max()) / 1e6, 2)},
        "comparison_median_over_combos_mbps": {"ofdma_variable": round(float(np.median(rmin_o)) / 1e6, 2),
                                               "hybrid_2x2": round(float(np.median(rmin)) / 1e6, 2),
                                               "mumimo_4x4": round(float(np.median(rmin_m)) / 1e6, 2)},
        "hybrid_beats_mumimo_pct_of_combos": round(float(np.mean(rmin > rmin_m) * 100), 2),
        "unique_placements": int(len(c4)),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / "hybrid_results.npy",
            {"combos": combos, "rmin": rmin, "rmean": rmean}, allow_pickle=True)
    (config.FIGURES_DIR / "11_hybrid_10cm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
