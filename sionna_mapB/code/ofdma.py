"""기준선 1: OFDMA만 사용 (부반송파 64개를 16개씩 대역 4개로 나누고, 대역마다 AMR 1대, BS는 MRC).

대역: B0 = 0~15, B1 = 16~31, B2 = 32~47, B3 = 48~63
AMR k가 대역 b를 받을 때 rate  R[k, b] = Δf · Σ_{n∈b} log₂(1 + P·‖h_k[n]‖² / σ²)   (MRC, 간섭 없음)
→ 그 AMR이 있는 셀 하나로만 정해지므로 셀 512개 × 대역 4개 표를 미리 만든다.

배정 방식
* adaptive: 0.1 s마다 24가지(4!) 배정 중 순간 최저 rate가 가장 큰 배정을 고른다.
* fixed   : AMR1→B0, AMR2→B1, AMR3→B2, AMR4→B3로 고정 (비교용).
R_min = 이동 전체 시간 동안 순간 최저 rate의 최솟값. 14만 경로 조합 전부에 대해 계산한다.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

import config
from collision import AMRS
from fingerprint import load_or_build_fingerprint
from mumimo import NX, NY, SNR_SCALE, cell_sequences
from path_planner import load_path_sets

N_BANDS = 4
BAND_SIZE = config.N_SC // N_BANDS
PERMS = np.array(list(itertools.permutations(range(N_BANDS))))   # (24, 4): PERMS[p, k] = AMR k의 대역
FIXED = np.arange(N_BANDS)


def band_rate_table(h_flat: np.ndarray) -> np.ndarray:
    """(512, 4) 셀 × 대역 MRC rate [bps]."""
    se = np.log2(1.0 + SNR_SCALE * np.sum(np.abs(h_flat) ** 2, axis=1))      # (512, 64)
    return config.DELTA_F_HZ * se.reshape(-1, N_BANDS, BAND_SIZE).sum(axis=2)


def best_assignment(table: np.ndarray, cells4: np.ndarray, chunk: int = 200_000):
    """배치 (U, 4)마다 24가지 배정 중 최저 rate 최대인 것: (최저 rate, 배정 번호)."""
    best_val = np.empty(len(cells4))
    best_idx = np.empty(len(cells4), dtype=np.int8)
    for a in range(0, len(cells4), chunk):
        c = cells4[a:a + chunk]
        r = table[c[:, None, :], PERMS[None, :, :]]                 # (u, 24, 4)
        worst = r.min(axis=2)
        best_idx[a:a + chunk] = worst.argmax(axis=1)
        best_val[a:a + chunk] = worst.max(axis=1)
    return best_val, best_idx


def evaluate_all(verbose: bool = True) -> dict:
    t0 = time.time()
    fp = load_or_build_fingerprint(verbose=False)
    h_flat = fp["H"].reshape(NY * NX, config.N_ANT_BS, config.N_SC).astype(np.complex128)
    table = band_rate_table(h_flat)
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = cell_sequences(paths, t)
    code = np.zeros((len(combos), len(t)), dtype=np.int64)
    for k, amr in enumerate(AMRS):
        code |= seq[amr][combos[:, k]].astype(np.int64) << (9 * (3 - k))
    uniq, inverse = np.unique(code, return_inverse=True)
    inverse = inverse.reshape(code.shape)
    cells4 = np.stack([(uniq >> (9 * (3 - k))) & 511 for k in range(4)], axis=1)
    adapt_u, _ = best_assignment(table, cells4)
    fixed_u = table[cells4, FIXED[None, :]].min(axis=1)
    adapt_t, fixed_t = adapt_u[inverse], fixed_u[inverse]
    if verbose:
        print(f"{len(combos):,} combos, {len(uniq):,} unique placements, {time.time() - t0:.0f} s", flush=True)
    return {"t": t, "seq": seq, "combos": combos, "paths": paths, "table": table,
            "rmin_adapt": adapt_t.min(axis=1), "rmin_fixed": fixed_t.min(axis=1),
            "mean_adapt": adapt_t.mean(axis=1), "runtime_s": time.time() - t0}


def trace_combo(res: dict, combo) -> dict:
    cells = np.stack([res["seq"][amr][combo[k]] for k, amr in enumerate(AMRS)], axis=1)   # (T, 4)
    val, idx = best_assignment(res["table"], cells)
    bands = PERMS[idx]                                                   # (T, 4)
    rates = res["table"][cells, bands]                                   # (T, 4)
    all_bands = res["table"][cells]                                      # (T, 4 AMR, 4 band)
    return {"cells": cells, "bands": bands, "rates": rates, "all_bands": all_bands,
            "fixed": res["table"][cells, FIXED[None, :]]}


def plot(res: dict, best, out_path, nx: int = NX, res_m: float = config.GRID_RES_M, tag: str = "50 cm map") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    t = res["t"]
    tr = trace_combo(res, best)
    tw = int(np.argmin(tr["rates"].min(axis=1)))
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.55], width_ratios=[1, 1.35])
    ax = fig.add_subplot(gs[0, 0])
    for ob in config.OBSTACLES:
        cx, cy = ob.center_xy_m
        if ob.kind == "box":
            dx, dy, _ = ob.size_xyz_m
            ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray"))
        else:
            ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray"))
    for k, amr in enumerate(AMRS):
        xy = np.asarray(res["paths"][amr + "_xy"][best[k]], float)
        ax.plot(xy[:, 0], xy[:, 1], color=colors[k], lw=2, label=f"{amr} path #{best[k]}")
        ax.plot(*xy[-1], "*", color=colors[k], ms=14, mec="k")
        r, c = divmod(int(tr["cells"][tw, k]), nx)
        ax.add_patch(Circle(((c + 0.5) * res_m, (r + 0.5) * res_m),
                            config.AMR_RADIUS_M, fc=colors[k], ec="k", alpha=0.8))
    ax.plot(*config.BS_CENTER_M[:2], "k^", ms=10)
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title(f"best paths, R_min {tr['rates'].min() / 1e6:.1f} Mbps\n(circles = positions at bottleneck t={t[tw]:.1f} s)")

    ax = fig.add_subplot(gs[0, 1])
    rmin_a, rmin_f = res["rmin_adapt"] / 1e6, res["rmin_fixed"] / 1e6
    bins = np.linspace(min(rmin_a.min(), rmin_f.min()), rmin_a.max(), 60)
    ax.hist(rmin_f, bins=bins, alpha=0.6, label=f"fixed bands (best {rmin_f.max():.1f})")
    ax.hist(rmin_a, bins=bins, alpha=0.6, label=f"adaptive band choice every 0.1 s (best {rmin_a.max():.1f})")
    ax.set_xlabel("R_min [Mbps]"); ax.set_ylabel("# path combinations"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"R_min over all {len(rmin_a):,} collision-free path combinations")

    ax = fig.add_subplot(gs[1, :])
    for k, amr in enumerate(AMRS):
        ax.plot(t, tr["rates"][:, k] / 1e6, color=colors[k], lw=1.4)
        ax.plot(t, tr["fixed"][:, k] / 1e6, color=colors[k], lw=0.8, ls=":")
    ax.plot(t, tr["rates"].min(axis=1) / 1e6, "k", lw=2.2)
    ax.plot(t, tr["fixed"].min(axis=1) / 1e6, "k", lw=1.2, ls=":")
    ax.axvline(t[tw], color="k", ls="--", lw=0.8)
    handles = [Line2D([], [], color=colors[k], lw=1.4, label=f"{amr}") for k, amr in enumerate(AMRS)]
    handles += [Line2D([], [], color="k", lw=2.2, label="min over AMRs (adaptive)"),
                Line2D([], [], color="gray", ls=":", label="dotted: fixed bands AMR1→B0 … AMR4→B3"),
                Line2D([], [], color="k", ls="--", lw=0.8, label="bottleneck time")]
    ax.legend(handles=handles, fontsize=8, ncol=7, loc="lower center")
    ax.set_ylabel("rate [Mbps]"); ax.grid(alpha=0.3)
    ax.set_title("rates along the best paths (16 subcarriers each, MRC)")

    ax = fig.add_subplot(gs[2, :], sharex=ax)
    for k, amr in enumerate(AMRS):
        ax.step(t, tr["bands"][:, k] + (k - 1.5) * 0.08, where="post", color=colors[k], lw=1.5, label=amr)
    ax.set_yticks(range(4), ["B0 (0-15)", "B1 (16-31)", "B2 (32-47)", "B3 (48-63)"])
    ax.set_xlabel("time [s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_title("band chosen for each AMR (adaptive)")
    fig.suptitle("Baseline 1: OFDMA only (4 bands x 16 subcarriers, 1 AMR per band, MRC) — channel: " + tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    res = evaluate_all()
    combos = res["combos"]
    ia, iff = int(np.argmax(res["rmin_adapt"])), int(np.argmax(res["rmin_fixed"]))
    best = combos[ia]
    tr = trace_combo(res, best)
    switches = int((np.diff(tr["bands"], axis=0) != 0).any(axis=1).sum())
    summary = {
        "scheme": "OFDMA only, 4 bands x 16 subcarriers, MRC, adaptive band permutation every 0.1 s",
        "n_combos_evaluated": int(len(combos)), "runtime_s": round(res["runtime_s"], 1),
        "adaptive": {"best_paths": dict(zip(AMRS, map(int, best))),
                     "rmin_mbps": round(float(res["rmin_adapt"][ia]) / 1e6, 2),
                     "mean_of_instant_min_mbps": round(float(res["mean_adapt"][ia]) / 1e6, 2),
                     "mean_rate_per_amr_mbps": dict(zip(AMRS, np.round(tr["rates"].mean(axis=0) / 1e6, 2).tolist())),
                     "band_switch_count": switches,
                     "median_over_combos_mbps": round(float(np.median(res["rmin_adapt"])) / 1e6, 2)},
        "fixed": {"best_paths": dict(zip(AMRS, map(int, combos[iff]))),
                  "rmin_mbps": round(float(res["rmin_fixed"][iff]) / 1e6, 2),
                  "rmin_on_adaptive_best_paths_mbps": round(float(res["rmin_fixed"][ia]) / 1e6, 2),
                  "median_over_combos_mbps": round(float(np.median(res["rmin_fixed"])) / 1e6, 2)},
    }
    np.save(config.PROJECT_ROOT / "code" / "ofdma_results.npy",
            {"combos": combos, "rmin_adapt_bps": res["rmin_adapt"], "rmin_fixed_bps": res["rmin_fixed"]}, allow_pickle=True)
    (config.FIGURES_DIR / "05_ofdma_summary.json").write_text(json.dumps(summary, indent=2))
    plot(res, best, config.FIGURES_DIR / "05_ofdma_best.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
