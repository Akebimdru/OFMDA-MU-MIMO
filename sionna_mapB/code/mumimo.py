"""기준선 2: MU-MIMO만 사용 (AMR 4대가 부반송파 64개 전체를 같이 쓰고, BS가 4×4 ZF).

업링크, 부반송파 n마다  y = H x + z,  H = [h_1 h_2 h_3 h_4]  (BS 안테나 4 × AMR 4)
ZF 수신  r = H⁻¹ y = x + H⁻¹ z
AMR k의 SINR = |x_k|² / E|(H⁻¹z)_k|² = P / (σ² · [(Hᴴ H)⁻¹]_kk)
AMR k의 순간 rate = Δf · Σ_n log₂(1 + SINR_k[n])

* 채널 h_k는 그 시각 AMR k가 있는 셀의 fingerprint (가우시안 잡음 3종 포함).
  ZF는 위상 관계가 중요해서 셀 사이 보간은 하지 않고, 로봇이 들어 있는 셀의 채널을 그대로 쓴다.
* 시각은 0.1 s 간격, 4대 동시 출발, 도착한 AMR은 목적지에 멈춘 채 계속 송신한다.
* R_min (조합의 성능) = 이동 전체 시간 동안 4대 중 가장 낮은 순간 rate의 최솟값 (이전 프로젝트와 같은 정의).
* 같은 (셀1, 셀2, 셀3, 셀4) 배치는 한 번만 계산한다.
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
from collision import AMRS
from fingerprint import load_or_build_fingerprint
from path_planner import load_path_sets

NX = int(round(config.ROOM_X_M / config.GRID_RES_M))
NY = int(round(config.ROOM_Y_M / config.GRID_RES_M))
SNR_SCALE = config.tx_power_w() / config.noise_power_w()


def _point_to_cell(xy: np.ndarray) -> np.ndarray:
    col = np.clip(np.floor(xy[:, 0] / config.GRID_RES_M).astype(int), 0, NX - 1)
    row = np.clip(np.floor(xy[:, 1] / config.GRID_RES_M).astype(int), 0, NY - 1)
    return row * NX + col


def cell_sequences(paths: dict, t: np.ndarray) -> dict[str, np.ndarray]:
    """AMR별 (26, T) 셀 번호. 출발/도착 꼭짓점에서는 경로가 붙은 셀(방 안쪽)을 쓴다."""
    out = {}
    for amr in AMRS:
        seqs = []
        for cells, xy in zip(paths[amr], paths[amr + "_xy"]):
            xy = np.asarray(xy, float)
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            s = np.minimum(t * config.AMR_SPEED_MPS, cum[-1])
            pos = np.c_[np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])]
            idx = _point_to_cell(pos)
            first, last = cells[0], cells[-1]
            idx[s <= seg[0]] = first[0] * NX + first[1]            # 꼭짓점 → 첫 셀 구간
            idx[s >= cum[-1] - seg[-1]] = last[0] * NX + last[1]   # 마지막 셀 → 도착 꼭짓점 구간
            seqs.append(idx)
        out[amr] = np.stack(seqs)
    return out


def zf_rates_bps(h_flat: np.ndarray, cells4: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """cells4 (U, 4) 배치마다 AMR 4대의 ZF rate [bps], (U, 4)."""
    out = np.empty((len(cells4), 4))
    for a in range(0, len(cells4), chunk):
        c = cells4[a:a + chunk]
        H = np.transpose(h_flat[c], (0, 3, 2, 1))                   # (u, 64, ant, user)
        G = np.conj(np.swapaxes(H, -1, -2)) @ H                     # Hᴴ H
        ginv_diag = np.real(np.diagonal(np.linalg.inv(G), axis1=-2, axis2=-1))   # (u, 64, 4)
        sinr = SNR_SCALE / np.maximum(ginv_diag, 1e-300)
        out[a:a + chunk] = config.DELTA_F_HZ * np.log2(1.0 + sinr).sum(axis=1)
    return out


def mrc_single_user_rates_bps(h_flat: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """비교용: 그 AMR 혼자 64개 부반송파를 다 쓰고 MRC할 때 rate (간섭 없음)."""
    snr = SNR_SCALE * np.sum(np.abs(h_flat[cells]) ** 2, axis=1)   # (n, 64)
    return config.DELTA_F_HZ * np.log2(1.0 + snr).sum(axis=1)


def evaluate_all(verbose: bool = True) -> dict:
    t0 = time.time()
    fp = load_or_build_fingerprint(verbose=False)
    h_flat = fp["H"].reshape(NY * NX, config.N_ANT_BS, config.N_SC).astype(np.complex128)
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = cell_sequences(paths, t)

    # 조합 × 시각의 4-셀 배치를 정수 하나로 묶어서 중복 제거
    code = np.zeros((len(combos), len(t)), dtype=np.int64)
    for k, amr in enumerate(AMRS):
        code |= seq[amr][combos[:, k]].astype(np.int64) << (9 * (3 - k))
    uniq, inverse = np.unique(code, return_inverse=True)
    cells4 = np.stack([(uniq >> (9 * (3 - k))) & 511 for k in range(4)], axis=1)
    if verbose:
        print(f"{len(combos):,} combos x {len(t)} steps -> {len(uniq):,} unique placements ({time.time() - t0:.0f} s)", flush=True)
    rates_u = zf_rates_bps(h_flat, cells4)                          # (U, 4)
    worst_u = rates_u.min(axis=1)
    inverse = inverse.reshape(code.shape)
    worst_t = worst_u[inverse]                                      # (C, T): 순간 최저 rate
    rmin = worst_t.min(axis=1)
    mean_worst = worst_t.mean(axis=1)
    if verbose:
        print(f"ZF done ({time.time() - t0:.0f} s)", flush=True)
    return {"t": t, "seq": seq, "combos": combos, "rmin": rmin, "mean_worst": mean_worst,
            "h_flat": h_flat, "paths": paths, "runtime_s": time.time() - t0}


def trace_combo(res: dict, combo: np.ndarray) -> dict:
    cells = np.stack([res["seq"][amr][combo[k]] for k, amr in enumerate(AMRS)], axis=1)   # (T, 4)
    zf = zf_rates_bps(res["h_flat"], cells)
    mrc = np.stack([mrc_single_user_rates_bps(res["h_flat"], cells[:, k]) for k in range(4)], axis=1)
    return {"cells": cells, "zf": zf, "mrc": mrc}


def plot(res: dict, best: np.ndarray, ref: np.ndarray, out_path, nx: int = NX, res_m: float = config.GRID_RES_M, tag: str = "50 cm map") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    t = res["t"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    for col, (combo, name) in enumerate(((best, "best R_min"), (ref, "fastest collision-free"))):
        ax = fig.add_subplot(gs[0, col])
        for ob in config.OBSTACLES:
            cx, cy = ob.center_xy_m
            if ob.kind == "box":
                dx, dy, _ = ob.size_xyz_m
                ax.add_patch(Rectangle((cx - dx / 2, cy - dy / 2), dx, dy, fc="dimgray"))
            else:
                ax.add_patch(Circle((cx, cy), ob.radius_m, fc="dimgray"))
        for k, amr in enumerate(AMRS):
            xy = np.asarray(res["paths"][amr + "_xy"][combo[k]], float)
            ax.plot(xy[:, 0], xy[:, 1], color=colors[k], lw=2, label=f"{amr} path #{combo[k]}")
            ax.plot(*xy[0], "o", color=colors[k], ms=8)
            ax.plot(*xy[-1], "*", color=colors[k], ms=14, mec="k")
        tr = trace_combo(res, combo)
        tw = int(np.argmin(tr["zf"].min(axis=1)))
        for k in range(4):
            r, c = divmod(int(tr["cells"][tw, k]), nx)
            ax.add_patch(Circle(((c + 0.5) * res_m, (r + 0.5) * res_m),
                                config.AMR_RADIUS_M, fc=colors[k], ec="k", alpha=0.8))
        ax.plot(*config.BS_CENTER_M[:2], "k^", ms=10)
        ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M); ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        ax.set_title(f"{name}: R_min {res['rmin_of'](combo) / 1e6:.1f} Mbps "
                     f"(filled circles = positions at the bottleneck t={t[tw]:.1f} s)")
        ax2 = fig.add_subplot(gs[1, col])
        for k, amr in enumerate(AMRS):
            ax2.plot(t, tr["zf"][:, k] / 1e6, color=colors[k], lw=1.4, label=f"{amr} ZF")
            ax2.plot(t, tr["mrc"][:, k] / 1e6, color=colors[k], lw=0.8, ls=":", alpha=0.8)
        ax2.plot(t, tr["zf"].min(axis=1) / 1e6, "k", lw=2.2, label="min over AMRs")
        ax2.axvline(t[tw], color="k", ls="--", lw=0.8)
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("rate [Mbps]"); ax2.grid(alpha=0.3)
        ax2.set_title("solid: MU-MIMO ZF (4 users share 64 subcarriers); dotted: same AMR alone with MRC")
        from matplotlib.lines import Line2D
        h2, l2 = ax2.get_legend_handles_labels()
        h2 += [Line2D([], [], color="gray", ls=":", label="dotted: same AMR alone (MRC, upper bound)"),
               Line2D([], [], color="k", ls="--", lw=0.8, label="bottleneck time")]
        ax2.legend(handles=h2, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle("Baseline 2: MU-MIMO only (4x4 ZF on the full band) — channel: " + tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    res = evaluate_all()
    combos, rmin = res["combos"], res["rmin"]
    lookup = {tuple(c): v for c, v in zip(map(tuple, combos), rmin)}
    res["rmin_of"] = lambda c: lookup[tuple(int(v) for v in c)]
    best = combos[np.argmax(rmin)]
    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in res["paths"][amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0)
    total = sum(lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS))
    ref = combos[np.argmin(makespan * 1000 + total)]
    tr_best = trace_combo(res, best)

    summary = {
        "scheme": "MU-MIMO only, 4x4 ZF, all 64 subcarriers shared",
        "n_combos_evaluated": int(len(combos)), "runtime_s": round(res["runtime_s"], 1),
        "best": {"paths": dict(zip(AMRS, map(int, best))), "rmin_mbps": round(float(rmin.max()) / 1e6, 2),
                 "mean_of_instant_min_mbps": round(float(res["mean_worst"][np.argmax(rmin)]) / 1e6, 2),
                 "mean_rate_per_amr_mbps": dict(zip(AMRS, np.round(tr_best["zf"].mean(axis=0) / 1e6, 2).tolist())),
                 "makespan_s": round(float(makespan[np.argmax(rmin)]) / config.AMR_SPEED_MPS, 1)},
        "fastest_collision_free": {"paths": dict(zip(AMRS, map(int, ref))),
                                   "rmin_mbps": round(float(res["rmin_of"](ref)) / 1e6, 2),
                                   "makespan_s": round(float(makespan.min()) / config.AMR_SPEED_MPS, 1)},
        "rmin_distribution_mbps": {q: round(float(np.percentile(rmin, p)) / 1e6, 2)
                                   for q, p in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))},
        "zf_loss_vs_single_user_mrc_best_combo_mbps": round(float((tr_best["mrc"] - tr_best["zf"]).mean()) / 1e6, 2),
    }
    np.save(config.PROJECT_ROOT / "code" / "mumimo_results.npy",
            {"combos": combos, "rmin_bps": rmin, "mean_worst_bps": res["mean_worst"]}, allow_pickle=True)
    (config.FIGURES_DIR / "04_mumimo_summary.json").write_text(json.dumps(summary, indent=2))
    plot(res, best, ref, config.FIGURES_DIR / "04_mumimo_best.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
