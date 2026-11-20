"""검증용: 연속 대역 + 순서 24가지에서 순간 R_min의 정확한 최적 (목표 rate 이분 탐색).

순서가 정해지면, 목표 R에 대해 왼쪽부터 각 AMR이 R을 넘길 때까지 최소 개수만 가져가는 탐욕이
'R 달성 가능?'을 정확히 판정한다 (남는 부반송파는 마지막 AMR에게). 이 판정을 R에 대해 이분 탐색.
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

import config
from collision import AMRS
from ofdma_var import ORDERS, balance, fixed16, sc_prefix_table, setup
from fine_eval import placements

N_BISECT = 24          # 0~400 Mbps 범위를 24번 → 약 24 bps 해상도


def _first_reach(prefix, pts, start, need):
    """prefix[pts, e] >= need 를 만족하는 가장 작은 e (start < e <= 64), 없으면 65. 이진 탐색 7단계."""
    lo, hi = start + 1, np.full_like(start, 65)
    for _ in range(7):
        mid = (lo + hi) // 2
        ok = (mid <= 64) & (prefix[pts, np.minimum(mid, 64)] >= need)
        act = lo < hi
        hi = np.where(act & ok, mid, hi)
        lo = np.where(act & ~ok, mid + 1, lo)
    return hi


def feasible(prefix, pts, R):
    """pts (m, 4) 위치별 점, R (m,) → 달성 가능 여부."""
    c = np.zeros(len(R), int)
    for j in range(3):
        need = prefix[pts[:, j], c] + R
        c = np.minimum(_first_reach(prefix, pts[:, j], c, need), 64)
    return prefix[pts[:, 3], 64] - prefix[pts[:, 3], c] >= R


def exact(prefix, cells4):
    u = len(cells4)
    pts = cells4[:, ORDERS].reshape(-1, 4)                          # (u*24, 4)
    lo = np.zeros(len(pts))
    hi = np.full(len(pts), 4e8)
    for _ in range(N_BISECT):
        mid = (lo + hi) / 2
        f = feasible(prefix, pts, mid)
        lo = np.where(f, mid, lo)
        hi = np.where(f, hi, mid)
    v = lo.reshape(u, 24)
    return v.max(axis=1), v.argmax(axis=1), v


def allocation(prefix, cells4, order, R):
    """주어진 순서·목표 R로 탐욕 분배한 AMR별 (시작, 크기)."""
    pts = cells4[np.arange(len(cells4))[:, None], ORDERS[order]]
    c = np.zeros(len(R), int)
    bounds = [c]
    for j in range(3):
        need = prefix[pts[:, j], c] + R
        c = np.minimum(_first_reach(prefix, pts[:, j], c, need), 64)
        bounds.append(c)
    bounds.append(np.full(len(R), 64))
    b = np.stack(bounds, axis=1)
    start, size = np.empty((len(R), 4), int), np.empty((len(R), 4), int)
    rows = np.arange(len(R))[:, None]
    start[rows, ORDERS[order]] = b[:, :-1]
    size[rows, ORDERS[order]] = np.diff(b, axis=1)
    return start, size


def brute_check(prefix, c4, n=300, seed=1):
    cuts = np.array(list(itertools.combinations(range(1, 64), 3)))
    B = np.concatenate([np.zeros((len(cuts), 1), int), cuts, np.full((len(cuts), 1), 64)], axis=1)
    s = c4[np.random.default_rng(seed).choice(len(c4), n, replace=False)]
    bf = np.empty(n)
    for i, cell in enumerate(s):
        P = prefix[cell]
        r = P[ORDERS[:, None, :], B[None, :, 1:]] - P[ORDERS[:, None, :], B[None, :, :-1]]
        bf[i] = r.min(axis=2).max()
    ex = exact(prefix, s)[0]
    hv = balance(prefix, s)[0]
    return {"n_samples": n, "max_abs_brute_minus_bisection_mbps": float(np.abs(bf - ex).max() / 1e6),
            "max_brute_minus_boundary_shift_mbps": float((bf - hv).max() / 1e6),
            "boundary_shift_within_0.01mbps_pct": float(np.mean(bf - hv < 1e4) * 100)}


def plot(t, paths, nx, res_m, best_e, tr_e, tr_h_on_e, rmin_e, rmin_h, rmin_f, gap_pl, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 0.8])
    # (a) 최적 경로
    ax = fig.add_subplot(gs[0, 0])
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray"))
        else:
            ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray"))
    tw = int(np.argmin(tr_e["rates"].min(axis=1)))
    for k, amr in enumerate(AMRS):
        xy = np.asarray(paths[amr + "_xy"][best_e[k]], float)
        ax.plot(xy[:, 0], xy[:, 1], color=colors[k], lw=2, label=f"{amr} #{best_e[k]}")
        ax.plot(*xy[-1], "*", color=colors[k], ms=12, mec="k")
        r, c = divmod(int(tr_e["cells"][tw, k]), nx)
        ax.add_patch(Circle(((c + 0.5) * res_m, (r + 0.5) * res_m), config.AMR_RADIUS_M, fc=colors[k], ec="k", alpha=0.8))
    ax.plot(*config.BS_CENTER_M[:2], "k^", ms=9)
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.set_title(f"(a) best paths, bisection (exact)\nR_min {rmin_e.max() / 1e6:.2f} Mbps, bottleneck t={t[tw]:.1f} s", fontsize=10)
    # (b) 조합별 산점도
    ax = fig.add_subplot(gs[0, 1])
    e, h = rmin_e / 1e6, rmin_h / 1e6
    ax.scatter(e, h, s=2, alpha=0.3)
    lim = [min(e.min(), h.min()) - 0.5, e.max() + 0.5]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("R_min, bisection (exact) [Mbps]"); ax.set_ylabel("R_min, boundary shifting [Mbps]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"(b) R_min per path combination ({len(e):,})\nsame value in {np.mean(e - h < 0.01) * 100:.2f}% of combinations", fontsize=10)
    # (c) 위치 조합별 순간 차이
    ax = fig.add_subplot(gs[0, 2])
    g = gap_pl / 1e6
    nz = g[g >= 0.01]
    ax.hist(nz, bins=50, color="tab:purple")
    ax.set_yscale("log"); ax.set_xlabel("instant R_min gap, exact - boundary shifting [Mbps]")
    ax.set_ylabel("# placements (gap >= 0.01 Mbps)"); ax.grid(alpha=0.3)
    ax.set_title(f"(c) per placement ({len(g):,} unique)\n{np.mean(g < 0.01) * 100:.2f}% identical, mean gap {g.mean() * 1e3:.2f} kbps, max {g.max():.2f} Mbps", fontsize=10)
    # (d) 히스토그램 3종
    ax = fig.add_subplot(gs[1, :])
    f = rmin_f / 1e6
    bins = np.linspace(f.min(), e.max() + 0.3, 90)
    ax.hist(f, bins=bins, alpha=0.5, label=f"16 SC each, best of 24 orders (best {f.max():.2f})")
    ax.hist(h, bins=bins, alpha=0.5, label=f"boundary shifting (best {h.max():.2f})")
    ax.hist(e, bins=bins, histtype="step", color="k", lw=1.3, label=f"bisection, exact contiguous optimum (best {e.max():.2f})")
    ax.set_xlabel("R_min [Mbps]"); ax.set_ylabel("# path combinations"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("(d) R_min over all collision-free path combinations", fontsize=10)
    # (e) 최적 경로에서의 순간 최저 rate
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t, tr_e["rates"].min(axis=1) / 1e6, "k", lw=1.8, label="bisection (exact)")
    ax.plot(t, tr_h_on_e / 1e6, color="tab:orange", lw=1.0, ls="--", label="boundary shifting")
    ax.plot(t, tr_e["frates"].min(axis=1) / 1e6, color="tab:blue", lw=0.8, ls=":", label="16 SC each")
    ax.axvline(t[tw], color="k", ls="--", lw=0.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel("instant min rate [Mbps]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    ax.set_title("(e) instantaneous min rate along the exact-best paths", fontsize=10)
    fig.suptitle("OFDMA variable band size: boundary shifting vs exact optimum (bisection on target rate, all 24 orders) — 10 cm map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    from fine_eval import placements
    from fine_map import FINE_RES_M
    t0 = time.time()
    h_flat, paths, combos, t, seq, nx = setup()
    prefix = sc_prefix_table(h_flat)
    c4, inv = placements(seq, combos)
    n = len(c4)
    ve, vh, vf = np.empty(n), np.empty(n), np.empty(n)
    oe = np.empty(n, np.int8)
    CH = 100_000
    for a in range(0, n, CH):
        c = c4[a:a + CH]
        ve[a:a + CH], oe[a:a + CH], _ = exact(prefix, c)
        vh[a:a + CH] = balance(prefix, c)[0]
        vf[a:a + CH] = fixed16(prefix, c)[0]
        done = min(a + CH, n); el = time.time() - t0
        print(f"{done:,}/{n:,}  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)
    gap_pl = np.maximum(ve - vh, 0.0)           # 이분 탐색 해상도(~24 bps)보다 작은 음수는 0
    def per_combo(v):
        x = v[inv]
        return x.min(axis=1), x.mean(axis=1)
    re, me = per_combo(ve); rh, mh = per_combo(vh); rf, _ = per_combo(vf)
    i_e = int(np.lexsort((me, re))[-1]); i_h = int(np.lexsort((mh, rh))[-1])
    best_e = combos[i_e]
    cells = np.stack([seq[amr][best_e[k]] for k, amr in enumerate(AMRS)], axis=1)
    v, o, _ = exact(prefix, cells)
    start, size = allocation(prefix, cells, o, v)
    rates = prefix[cells, start + size] - prefix[cells, start]
    fv, fo = fixed16(prefix, cells)
    from ofdma import PERMS
    fb = PERMS[fo]
    tr_e = {"cells": cells, "rates": rates, "sizes": size,
            "frates": prefix[cells, 16 * (fb + 1)] - prefix[cells, 16 * fb]}
    tr_h_on_e = balance(prefix, cells)[0]
    plot(t, paths, nx, FINE_RES_M, best_e, tr_e, tr_h_on_e, re, rh, rf, gap_pl,
         config.FIGURES_DIR / "10_ofdma_exact_vs_shift_10cm.png")
    print("brute-force spot check ...", flush=True)
    bc = brute_check(prefix, c4)
    tie = lambda r: int(np.isclose(r, r.max(), rtol=0, atol=1e3).sum())
    d = (re - rh) / 1e6
    summary = {
        "channel": "10 cm Sionna map, simultaneous departure",
        "exact_bisection": {"best_paths": dict(zip(AMRS, map(int, best_e))),
                            "rmin_mbps": round(float(re[i_e]) / 1e6, 3),
                            "combos_tied_at_best": tie(re),
                            "mean_of_instant_min_mbps": round(float(me[i_e]) / 1e6, 2),
                            "boundary_shift_on_same_paths_rmin_mbps": round(float(rh[i_e]) / 1e6, 3),
                            "band_size_range_per_amr": {amr: [int(size[:, k].min()), int(size[:, k].max())] for k, amr in enumerate(AMRS)},
                            "median_over_combos_mbps": round(float(np.median(re)) / 1e6, 2)},
        "boundary_shift": {"best_paths": dict(zip(AMRS, map(int, combos[i_h]))),
                           "rmin_mbps": round(float(rh[i_h]) / 1e6, 3),
                           "median_over_combos_mbps": round(float(np.median(rh)) / 1e6, 2),
                           "best_as_pct_of_exact_best": round(float(rh.max() / re.max() * 100), 3)},
        "fixed16_best_of_24_rmin_mbps": round(float(rf.max()) / 1e6, 2),
        "per_combination_gap_mbps": {"identical_pct(<0.01)": round(float(np.mean(d < 0.01) * 100), 2),
                                     "mean": round(float(d.mean()), 4), "max": round(float(d.max()), 3)},
        "per_placement_gap_mbps": {"identical_pct(<0.01)": round(float(np.mean(gap_pl < 1e4) * 100), 3),
                                   "mean": round(float(gap_pl.mean() / 1e6), 5), "max": round(float(gap_pl.max() / 1e6), 3)},
        "brute_force_spot_check": bc,
        "unique_placements": int(n),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / "ofdma_exact_results.npy",
            {"combos": combos, "exact_rmin": re, "exact_mean": me, "shift_rmin": rh, "fixed16_rmin": rf}, allow_pickle=True)
    (config.FIGURES_DIR / "10_ofdma_exact_vs_shift_10cm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bench":
        h_flat, paths, combos, t, seq, nx = setup()
        prefix = sc_prefix_table(h_flat)
        c4, inv = placements(seq, combos)
        s = c4[np.random.default_rng(0).choice(len(c4), 20_000, replace=False)]
        t0 = time.time(); ex = exact(prefix, s)[0]; dt = time.time() - t0
        print(f"20k exact {dt:.1f} s -> all {dt * len(c4) / 20_000 / 60:.1f} min")
    else:
        main()
