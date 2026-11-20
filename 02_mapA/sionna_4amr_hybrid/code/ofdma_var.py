"""기준선 1-2: OFDMA + 가변 대역 크기 (k1~k4, 합 64, 연속 대역, MRC).

알고리즘 (사용자 설계, 2026-11-07)
* 0.1 s마다, 대역 순서(왼쪽부터 어느 AMR이 오는지) 24가지 각각에 대해
  1) 16개씩 균등 분배에서 시작한다.
  2) rate가 가장 높은 AMR에게서 부반송파 1개를 떼어 가장 낮은 AMR에게 준다.
     대역이 연속이어야 하므로 두 AMR 사이 경계들을 연쇄적으로 한 칸씩 민다
     (가운데 AMR들은 1개 받고 1개 주므로 개수는 그대로).
  3) 옮긴 뒤 순간 R_min이 엄격히 커질 때만 받아들이고, 아니면 멈춘다 (R_min 단조 증가 → 반드시 종료).
* 24가지 중 순간 R_min이 가장 큰 순서를 고른다.
* 부반송파별 rate 누적합(prefix sum)으로 대역 rate는 뺄셈 한 번.
* 결과는 점 4개(배치)로만 정해지므로 고유 배치마다 한 번만 계산한다.

출력: figures/09_ofdma_var_10cm.png, 09_ofdma_var_10cm_summary.json, code/ofdma_var_results.npy
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import config
from collision import AMRS
from fine_eval import fine_sequences, load_fine, placements
from mumimo import SNR_SCALE
from ofdma import PERMS
from path_planner import load_path_sets

N_SC = config.N_SC
ORDERS = np.argsort(PERMS, axis=1)          # ORDERS[p, j] = 대역 위치 j에 놓이는 AMR
INIT_BOUNDS = np.array([0, 16, 32, 48, 64])
MAX_ITERS = 64


def sc_prefix_table(h_flat: np.ndarray) -> np.ndarray:
    """(점, 65) 부반송파별 MRC rate 누적합 [bps]."""
    se = np.log2(1.0 + SNR_SCALE * np.sum(np.abs(h_flat) ** 2, axis=1))      # (점, 64)
    return np.concatenate([np.zeros((len(se), 1)), np.cumsum(config.DELTA_F_HZ * se, axis=1)], axis=1)


def _rates(prefix, pts, b):
    """pts (u, 24, 4) 위치별 점, b (u, 24, 5) 경계 → (u, 24, 4) 위치별 rate."""
    return prefix[pts, b[..., 1:]] - prefix[pts, b[..., :-1]]


def balance(prefix: np.ndarray, cells4: np.ndarray):
    """배치 (u, 4)마다 (최선 순간 R_min, 순서 번호, AMR별 부반송파 수 (u,4), 반복 횟수 최대)."""
    u = len(cells4)
    pts = cells4[:, ORDERS]                                          # (u, 24, 4)
    b = np.broadcast_to(INIT_BOUNDS, (u, 24, 5)).copy()
    r = _rates(prefix, pts, b)
    cur = r.min(axis=2)
    active = np.ones((u, 24), bool)
    j = np.arange(1, 4)
    it = 0
    while active.any() and it < MAX_ITERS:
        it += 1
        ia = np.nonzero(active)
        ra, ba = r[ia], b[ia]                                        # (m, 4), (m, 5)
        pmax, pmin = ra.argmax(axis=1), ra.argmin(axis=1)
        size = np.diff(ba, axis=1)
        ok = size[np.arange(len(pmax)), pmax] > 1
        # pmax < pmin: 경계 pmax+1..pmin 을 왼쪽으로 (-1), pmax > pmin: 경계 pmin+1..pmax 를 오른쪽으로 (+1)
        lo, hi = np.minimum(pmax, pmin)[:, None], np.maximum(pmax, pmin)[:, None]
        shift = np.where((j > lo) & (j <= hi), np.where(pmax < pmin, -1, 1)[:, None], 0)
        nb = ba.copy()
        nb[:, 1:4] += shift
        nr = prefix[pts[ia], nb[:, 1:]] - prefix[pts[ia], nb[:, :-1]]
        new = nr.min(axis=1)
        acc = ok & (new > cur[ia])
        ai = tuple(x[acc] for x in ia)
        b[ai], r[ai], cur[ai] = nb[acc], nr[acc], new[acc]
        active[tuple(x[~acc] for x in ia)] = False
    best = cur.argmax(axis=1)
    sel = np.arange(u)
    sizes_pos = np.diff(b[sel, best], axis=1)                        # 위치별 크기
    sizes = np.empty_like(sizes_pos)
    sizes[sel[:, None], ORDERS[best]] = sizes_pos                    # AMR별 크기
    return cur[sel, best], best, sizes, it


def setup():
    meta, H, h_flat, remap, ny, nx = load_fine()
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = fine_sequences(paths, t, nx, ny, remap)
    return h_flat, paths, combos, t, seq, nx


def benchmark():
    t0 = time.time()
    h_flat, paths, combos, t, seq, nx = setup()
    prefix = sc_prefix_table(h_flat)
    print(f"setup {time.time() - t0:.1f} s", flush=True)
    t1 = time.time()
    c4, inv = placements(seq, combos)
    print(f"unique placements {len(c4):,} ({time.time() - t1:.1f} s)", flush=True)
    rng = np.random.default_rng(0)
    s = c4[rng.choice(len(c4), 50_000, replace=False)]
    t2 = time.time()
    val, best, sizes, it = balance(prefix, s)
    dt = time.time() - t2
    fixed = (prefix[s[:, None, :], (16 * (PERMS + 1))[None]] - prefix[s[:, None, :], (16 * PERMS)[None]]).min(axis=2).max(axis=1)
    print(f"50k placements: {dt:.1f} s, iters {it}, gain median {np.median(val - fixed) / 1e6:.2f} Mbps")
    print(f"estimate all: {dt * len(c4) / 50_000 / 60:.1f} min")


def fixed16(prefix, cells4):
    """기존 방식(16개 고정 + 24가지 배정 중 최선)의 순간 R_min과 배정 번호."""
    r = prefix[cells4[:, None, :], (16 * (PERMS + 1))[None]] - prefix[cells4[:, None, :], (16 * PERMS)[None]]
    w = r.min(axis=2)
    return w.max(axis=1), w.argmax(axis=1)


def run_all(prefix, c4, chunk=200_000):
    n = len(c4)
    val, order = np.empty(n), np.empty(n, np.int8)
    sizes, fval = np.empty((n, 4), np.int8), np.empty(n)
    iters = 0
    t0 = time.time()
    for a in range(0, n, chunk):
        v, o, s, it = balance(prefix, c4[a:a + chunk])
        val[a:a + chunk], order[a:a + chunk], sizes[a:a + chunk] = v, o, s
        fval[a:a + chunk] = fixed16(prefix, c4[a:a + chunk])[0]
        iters = max(iters, it)
        done = min(a + chunk, n)
        el = time.time() - t0
        print(f"{done:,}/{n:,} placements  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)
    return val, order, sizes, fval, iters


def trace(prefix, seq, combo):
    cells = np.stack([seq[amr][combo[k]] for k, amr in enumerate(AMRS)], axis=1)      # (T, 4)
    v, o, sizes, _ = balance(prefix, cells)
    # AMR별 대역 시작/끝: 순서대로 크기를 쌓는다
    order_amr = ORDERS[o]                                                              # (T, 4) 위치→AMR
    start = np.empty((len(o), 4), int)
    for ti in range(len(o)):
        c = 0
        for amr_k in order_amr[ti]:
            start[ti, amr_k] = c
            c += sizes[ti, amr_k]
    rates = prefix[cells, start + sizes] - prefix[cells, start]
    fv, fo = fixed16(prefix, cells)
    fb = PERMS[fo]
    frates = prefix[cells, 16 * (fb + 1)] - prefix[cells, 16 * fb]
    return {"cells": cells, "sizes": sizes, "start": start, "rates": rates, "frates": frates}


def plot(t, paths, seq, prefix, nx, best, rmin_var, rmin_fix, out_path, res_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    tr = trace(prefix, seq, best)
    tw = int(np.argmin(tr["rates"].min(axis=1)))
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8], width_ratios=[1, 1.35])
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
        ax.add_patch(Circle(((c + 0.5) * res_m, (r + 0.5) * res_m), config.AMR_RADIUS_M, fc=colors[k], ec="k", alpha=0.8))
    ax.plot(*config.BS_CENTER_M[:2], "k^", ms=10)
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title(f"best paths, R_min {tr['rates'].min() / 1e6:.1f} Mbps\n(circles = positions at bottleneck t={t[tw]:.1f} s)")

    ax = fig.add_subplot(gs[0, 1])
    a, f = rmin_var / 1e6, rmin_fix / 1e6
    bins = np.linspace(min(a.min(), f.min()), a.max(), 60)
    ax.hist(f, bins=bins, alpha=0.6, label=f"16 SC each, best of 24 orders (best {f.max():.1f})")
    ax.hist(a, bins=bins, alpha=0.6, label=f"variable band size k1..k4 (best {a.max():.1f})")
    ax.set_xlabel("R_min [Mbps]"); ax.set_ylabel("# path combinations"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"R_min over all {len(a):,} collision-free path combinations")

    ax = fig.add_subplot(gs[1, :])
    for k, amr in enumerate(AMRS):
        ax.plot(t, tr["rates"][:, k] / 1e6, color=colors[k], lw=1.4)
        ax.plot(t, tr["frates"][:, k] / 1e6, color=colors[k], lw=0.8, ls=":")
    ax.plot(t, tr["rates"].min(axis=1) / 1e6, "k", lw=2.2)
    ax.plot(t, tr["frates"].min(axis=1) / 1e6, "k", lw=1.2, ls=":")
    ax.axvline(t[tw], color="k", ls="--", lw=0.8)
    handles = [Line2D([], [], color=colors[k], lw=1.4, label=amr) for k, amr in enumerate(AMRS)]
    handles += [Line2D([], [], color="k", lw=2.2, label="min over AMRs (variable size)"),
                Line2D([], [], color="gray", ls=":", label="dotted: 16 SC each, same paths"),
                Line2D([], [], color="k", ls="--", lw=0.8, label="bottleneck time")]
    ax.legend(handles=handles, fontsize=8, ncol=7, loc="lower center")
    ax.set_ylabel("rate [Mbps]"); ax.grid(alpha=0.3)
    ax.set_title("rates along the best paths (MRC)")

    ax = fig.add_subplot(gs[2, :], sharex=ax)
    for k, amr in enumerate(AMRS):
        ax.step(t, tr["sizes"][:, k], where="post", color=colors[k], lw=1.5, label=amr)
    ax.axhline(16, color="k", ls=":", lw=0.8)
    ax.set_ylabel("# subcarriers k_i"); ax.set_xlabel("time [s]"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_title("band size given to each AMR (sum 64, contiguous bands)")
    fig.suptitle("Baseline 1b: OFDMA with variable band size (boundary shifting until R_min stops increasing) — channel: 10 cm map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return tr


def main():
    from fine_map import FINE_RES_M
    t0 = time.time()
    h_flat, paths, combos, t, seq, nx = setup()
    prefix = sc_prefix_table(h_flat)
    c4, inv = placements(seq, combos)
    print(f"{len(c4):,} unique placements", flush=True)
    val, order, sizes, fval, iters = run_all(prefix, c4)
    inst = val[inv]                                       # (combos, T)
    rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
    finst = fval[inv]
    frmin, frmean = finst.min(axis=1), finst.mean(axis=1)
    del inst, finst
    i_v = int(np.lexsort((rmean, rmin))[-1])
    i_f = int(np.lexsort((frmean, frmin))[-1])
    tie = lambda r: int(np.isclose(r, r.max(), rtol=0, atol=1e3).sum())
    best = combos[i_v]
    tr = plot(t, paths, seq, prefix, nx, best, rmin, frmin, config.FIGURES_DIR / "09_ofdma_var_10cm.png", FINE_RES_M)
    gain = rmin - frmin
    summary = {
        "channel": "10 cm Sionna map, simultaneous departure",
        "algorithm": "per 0.1 s and per band order (24): start 16 each, shift contiguous boundaries one SC from max-rate to min-rate AMR while instantaneous R_min strictly increases; best order",
        "variable_size": {"best_paths": dict(zip(AMRS, map(int, best))),
                          "rmin_mbps": round(float(rmin[i_v]) / 1e6, 2),
                          "combos_tied_at_best": tie(rmin),
                          "mean_of_instant_min_mbps": round(float(rmean[i_v]) / 1e6, 2),
                          "same_paths_16each_rmin_mbps": round(float(frmin[i_v]) / 1e6, 2),
                          "band_size_range_per_amr": {amr: [int(tr["sizes"][:, k].min()), int(tr["sizes"][:, k].max())] for k, amr in enumerate(AMRS)},
                          "median_over_combos_mbps": round(float(np.median(rmin)) / 1e6, 2)},
        "fixed16_best_of_24": {"best_paths": dict(zip(AMRS, map(int, combos[i_f]))),
                               "rmin_mbps": round(float(frmin[i_f]) / 1e6, 2),
                               "median_over_combos_mbps": round(float(np.median(frmin)) / 1e6, 2)},
        "gain_over_16each_same_paths_mbps": {"median": round(float(np.median(gain)) / 1e6, 2),
                                              "max": round(float(gain.max()) / 1e6, 2),
                                              "min": round(float(gain.min()) / 1e6, 2)},
        "unique_placements": int(len(c4)),
        "max_iterations": int(iters),
        "mumimo_rmin_mbps_ref": 145.17,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / "ofdma_var_results.npy",
            {"combos": combos, "rmin": rmin, "rmean": rmean, "fixed16_rmin": frmin}, allow_pickle=True)
    (config.FIGURES_DIR / "09_ofdma_var_10cm_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        benchmark()
    else:
        main()
