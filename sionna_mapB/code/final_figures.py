"""방식 4가지를 같은 조건(같은 컴퓨터, 10 cm 맵, 충돌 없는 142,347 조합)에서 다시 돌려
계산 시간을 재고, 방식마다 그림 1장씩 만든다.

1. OFDMA, 16개 고정 대역 (0.1 s마다 24가지 배정 중 최선)
2. OFDMA, 대역 크기 가변 (경계 밀기, 연속 대역 최적과 일치 확인됨)
3. MU-MIMO, 4×4 ZF (전 대역에 4대)
4. 제안: 하이브리드 2×2 (대역 2개 × 2대, 378가지 전수 비교)

계산 시간 = 고유 배치 계산 + 경로 조합별 R_min 집계 + 최적 조합 선택 (배치 목록 만들기 시간은 공통으로 더함).
출력: figures/12_final_{1..4}_*.png, figures/12_final_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import mumimo
from collision import AMRS
from fine_eval import placements
from fine_map import FINE_RES_M
from hybrid import OPTION_LABEL, best_hybrid, option_rates, pair_prefix
from ofdma import PERMS
from ofdma_var import ORDERS, balance, fixed16, sc_prefix_table, setup

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


# ---------------- 방식별: (고유 배치 → 순간 R_min) 과 (최적 조합 → AMR별 rate, 부가 정보) ----------------
def m_fixed(ctx, c4):
    return fixed16(ctx["prefix"], c4)[0]


def t_fixed(ctx, cells):
    _, idx = fixed16(ctx["prefix"], cells)
    b = PERMS[idx]
    r = ctx["prefix"][cells, 16 * (b + 1)] - ctx["prefix"][cells, 16 * b]
    return r, {"band index (0-15, 16-31, 32-47, 48-63)": b}


def m_var(ctx, c4):
    out = np.empty(len(c4))
    for a in range(0, len(c4), 200_000):
        out[a:a + 200_000] = balance(ctx["prefix"], c4[a:a + 200_000])[0]
    return out


def t_var(ctx, cells):
    v, o, sizes, _ = balance(ctx["prefix"], cells)
    start = np.empty_like(sizes, dtype=int)
    for ti in range(len(o)):
        c = 0
        for k in ORDERS[o[ti]]:
            start[ti, k] = c
            c += sizes[ti, k]
    r = ctx["prefix"][cells, start + sizes] - ctx["prefix"][cells, start]
    return r, {"# subcarriers k_i": sizes}


def m_mumimo(ctx, c4):
    return mumimo.zf_rates_bps(ctx["h"], c4).min(axis=1)


def t_mumimo(ctx, cells):
    return mumimo.zf_rates_bps(ctx["h"], cells), {}


def m_hybrid(ctx, c4):
    return best_hybrid(ctx["h"], c4)[0]


def t_hybrid(ctx, cells):
    v, o, k = best_hybrid(ctx["h"], cells)
    r = option_rates(pair_prefix(ctx["h"], cells))[np.arange(len(cells)), o, k - 1]
    return r, {"pairing / band order": o, "k (size of band A)": k}


METHODS = [
    ("1_ofdma_fixed16", "Baseline 1a: OFDMA, 4 fixed bands x 16 SC (best of 24 assignments every 0.1 s), MRC", m_fixed, t_fixed),
    ("2_ofdma_variable", "Baseline 1b: OFDMA, variable contiguous band sizes (boundary shifting), MRC", m_var, t_var),
    ("3_mumimo", "Baseline 2: MU-MIMO, all 4 AMRs on full band, 4x4 ZF", m_mumimo, t_mumimo),
    ("4_hybrid", "Proposed: hybrid, 2 bands x 2 AMRs, ZF within band (378 options every 0.1 s)", m_hybrid, t_hybrid),
]


def plot(name, title, t, paths, nx, best, rates, extra, calc_s, lengths, n_combos, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    inst_min = rates.min(axis=1)
    rmin = inst_min.min()
    tw = int(np.argmin(inst_min))
    travel = [lengths[amr][best[k]] / config.AMR_SPEED_MPS for k, amr in enumerate(AMRS)]
    makespan = max(travel)
    n_extra = len(extra)
    fig = plt.figure(figsize=(16, 10 + 1.8 * n_extra))
    gs = fig.add_gridspec(2 + n_extra, 2, height_ratios=[1.25, 1.1] + [0.5] * n_extra, width_ratios=[1.15, 1])

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
        ax.plot(xy[:, 0], xy[:, 1], color=COLORS[k], lw=2.2, label=f"{amr} path #{best[k]}")
        ax.plot(*xy[0], "o", color=COLORS[k], ms=8, mec="k")
        ax.plot(*xy[-1], "*", color=COLORS[k], ms=15, mec="k")
        r, c = divmod(int(tr_cells[tw, k]), nx)
        ax.add_patch(Circle(((c + 0.5) * FINE_RES_M, (r + 0.5) * FINE_RES_M), config.AMR_RADIUS_M,
                            fc=COLORS[k], ec="k", alpha=0.5, ls="--"))
    ax.plot(*config.BS_CENTER_M[:2], "k^", ms=11, label="BS")
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    ax.set_title("selected paths (o start, * goal, faded circles = positions at bottleneck)", fontsize=10)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    rows = [("R_min (min over time & AMRs)", f"{rmin / 1e6:.2f} Mbps"),
            ("bottleneck", f"t = {t[tw]:.1f} s, {AMRS[int(rates[tw].argmin())]}"),
            ("time-mean of instantaneous min rate", f"{inst_min.mean() / 1e6:.2f} Mbps"),
            ("total travel time (makespan)", f"{makespan:.1f} s"),
            *[(f"  {amr}: path #{best[k]}", f"{lengths[amr][best[k]]:.2f} m, {travel[k]:.1f} s") for k, amr in enumerate(AMRS)],
            ("total computation time", f"{calc_s / 60:.1f} min ({calc_s:.0f} s)"),
            ("  path combinations evaluated", f"{n_combos:,} (collision-free)")]
    y = 0.95
    for a, b in rows:
        bold = not a.startswith("  ")
        ax.text(0.0, y, a, fontsize=12 if bold else 11, weight="bold" if bold else "normal", transform=ax.transAxes)
        ax.text(1.0, y, b, fontsize=12 if bold else 11, ha="right", transform=ax.transAxes)
        y -= 0.095
    ax.set_title("summary", fontsize=11)

    ax = fig.add_subplot(gs[1, :])
    for k, amr in enumerate(AMRS):
        ax.plot(t, rates[:, k] / 1e6, color=COLORS[k], lw=1.2)
    ax.plot(t, inst_min / 1e6, "k", lw=2.2)
    ax.axhline(rmin / 1e6, color="crimson", ls="--", lw=1.2)
    ax.axvline(t[tw], color="k", ls=":", lw=0.8)
    for k in range(4):
        ax.axvline(travel[k], color=COLORS[k], ls=":", lw=0.9, alpha=0.8)
    handles = [Line2D([], [], color=COLORS[k], lw=1.2, label=f"R_{k + 1}(t)") for k in range(4)]
    handles += [Line2D([], [], color="k", lw=2.2, label="min_k R_k(t)"),
                Line2D([], [], color="crimson", ls="--", label=f"R_min = {rmin / 1e6:.2f} Mbps"),
                Line2D([], [], color="gray", ls=":", label="arrival times / bottleneck")]
    ax.legend(handles=handles, fontsize=8, ncol=7, loc="lower center")
    ax.set_xlabel("time [s]"); ax.set_ylabel("rate [Mbps]"); ax.grid(alpha=0.3)
    ax.set_title("rate of each AMR over time and instantaneous minimum", fontsize=10)
    lo = min(0.9 * rmin / 1e6, ax.get_ylim()[0])
    ax.set_ylim(bottom=lo - 0.12 * (ax.get_ylim()[1] - lo))

    for i, (lab, arr) in enumerate(extra.items()):
        ax2 = fig.add_subplot(gs[2 + i, :], sharex=ax)
        if arr.ndim == 2:
            for k, amr in enumerate(AMRS):
                ax2.step(t, arr[:, k] + (k - 1.5) * 0.06, where="post", color=COLORS[k], lw=1.3, label=amr)
            ax2.legend(fontsize=8, ncol=4, loc="upper right")
        else:
            ax2.step(t, arr, where="post", color="k", lw=1.3)
        if lab.startswith("pairing"):
            ax2.set_yticks(range(6), OPTION_LABEL, fontsize=8)
        ax2.set_title(lab, fontsize=10); ax2.grid(alpha=0.3)
        ax2.set_xlabel("time [s]")
    fig.suptitle(title + " — 10 cm Sionna map", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return rmin, tw, makespan, travel, inst_min.mean()


def main():
    global tr_cells
    h_flat, paths, combos, t, seq, nx = setup()
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    t0 = time.time()
    c4, inv = placements(seq, combos)
    t_place = time.time() - t0
    print(f"placements {len(c4):,} in {t_place:.1f} s", flush=True)
    summary = {"common": {"unique_placements": int(len(c4)), "placement_build_s": round(t_place, 1),
                          "path_combinations": int(len(combos)), "cpu": "4-core cloud container, numpy"}}
    for name, title, mfun, tfun in METHODS:
        t0 = time.time()
        ctx = {"h": h_flat, "prefix": sc_prefix_table(h_flat)}
        val = mfun(ctx, c4)
        inst = val[inv]
        rmin, rmean = inst.min(axis=1), inst.mean(axis=1)
        del inst
        i_b = int(np.lexsort((rmean, rmin))[-1])
        calc = time.time() - t0 + t_place
        best = combos[i_b]
        tr_cells = np.stack([seq[amr][best[k]] for k, amr in enumerate(AMRS)], axis=1)
        rates, extra = tfun(ctx, tr_cells)
        assert abs(rates.min(axis=1).min() - rmin[i_b]) < 1e3, (name, rates.min(), rmin[i_b])
        r, tw, mk, travel, mean_min = plot(name, title, t, paths, nx, best, rates, extra, calc, lengths, len(combos),
                                           config.FIGURES_DIR / f"12_final_{name}.png")
        summary[name] = {"title": title, "computation_time_s": round(calc, 1),
                         "best_paths": dict(zip(AMRS, map(int, best))),
                         "rmin_mbps": round(r / 1e6, 2), "bottleneck_time_s": round(float(t[tw]), 1),
                         "bottleneck_amr": AMRS[int(rates[tw].argmin())],
                         "mean_of_instant_min_mbps": round(float(mean_min) / 1e6, 2),
                         "makespan_s": round(mk, 1),
                         "travel_time_s": dict(zip(AMRS, [round(x, 1) for x in travel])),
                         "combos_tied_at_best": int(np.isclose(rmin, rmin.max(), rtol=0, atol=1e3).sum()),
                         "median_over_combos_mbps": round(float(np.median(rmin)) / 1e6, 2)}
        print(name, json.dumps(summary[name]), flush=True)
    (config.FIGURES_DIR / "12_final_summary.json").write_text(json.dumps(summary, indent=2))


tr_cells = None

if __name__ == "__main__":
    main()
