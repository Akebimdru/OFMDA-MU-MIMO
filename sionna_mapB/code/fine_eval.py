"""10 cm 맵으로 두 기준선 다시 평가 + 50 cm 맵과 비교.

* 로봇 위치(0.1 s마다 5 cm씩 이동)를 그 위치가 들어 있는 10 cm 칸의 채널로 바꾼다.
* OFDMA: 점 × 대역 rate 표가 작아서 14만 조합 전부 다시 계산한다.
* MU-MIMO: 4×4 ZF는 배치가 많아져서, 50 cm 결과 상위 3,000개 + 무작위 3,000개 조합만 다시 계산한다.
  (무작위 표본으로 50 cm 순위와 10 cm 순위가 얼마나 일치하는지도 본다.)
그림: 06_fine_map_snr.png (50 cm vs 10 cm SNR 맵, 1 m 확대), 07_fine_vs_coarse.png (기준선 결과 비교)
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
from collision import AMRS
from fine_map import FINE_H_CLEAN_PATH, FINE_H_PATH, FINE_META_PATH, FINE_RES_M
from mumimo import SNR_SCALE, zf_rates_bps
from ofdma import FIXED, band_rate_table, best_assignment
from path_planner import load_path_sets

RNG = np.random.default_rng(config.RANDOM_SEED)


def load_fine():
    meta = np.load(FINE_META_PATH)
    if FINE_H_PATH.exists():
        H = np.load(FINE_H_PATH)
    else:  # Desktop 복사본은 25 MB 제한 때문에 두 조각으로 나뉘어 있다
        H = np.concatenate([np.load(FINE_H_PATH.with_name("fine_H_rows00_39.npy")),
                            np.load(FINE_H_PATH.with_name("fine_H_rows40_79.npy"))])
    ny, nx = H.shape[:2]
    h_flat = H.reshape(ny * nx, config.N_ANT_BS, config.N_SC).astype(np.complex128)
    # 장애물 안쪽(채널 없음) 점은 가장 가까운 빈 점으로 대신한다 (로봇 중심이 들어갈 일은 거의 없음)
    empty = np.abs(h_flat).sum(axis=(1, 2)) == 0
    remap = np.arange(ny * nx)
    if empty.any():
        free = np.flatnonzero(~empty)
        fy, fx = np.divmod(free, nx)
        for i in np.flatnonzero(empty):
            iy, ix = divmod(i, nx)
            remap[i] = free[np.argmin((fy - iy) ** 2 + (fx - ix) ** 2)]
    return meta, H, h_flat, remap, ny, nx


def fine_sequences(paths, t, nx, ny, remap):
    out = {}
    for amr in AMRS:
        seqs = []
        for xy in paths[amr + "_xy"]:
            xy = np.asarray(xy, float)
            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            s = np.minimum(t * config.AMR_SPEED_MPS, cum[-1])
            px, py = np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])
            col = np.clip(np.floor(px / FINE_RES_M + 1e-9).astype(int), 0, nx - 1)
            row = np.clip(np.floor(py / FINE_RES_M + 1e-9).astype(int), 0, ny - 1)
            seqs.append(remap[row * nx + col])
        out[amr] = np.stack(seqs)
    return out


def placements(seq, combos):
    code = np.zeros((len(combos), seq["AMR1"].shape[1]), dtype=np.int64)
    for k, amr in enumerate(AMRS):
        code |= seq[amr][combos[:, k]].astype(np.int64) << (14 * (3 - k))
    uniq, inv = np.unique(code, return_inverse=True)
    cells4 = np.stack([(uniq >> (14 * (3 - k))) & (2 ** 14 - 1) for k in range(4)], axis=1)
    return cells4, inv.reshape(code.shape)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.time()
    meta, H, h_flat, remap, ny, nx = load_fine()
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = fine_sequences(paths, t, nx, ny, remap)

    # ---- OFDMA: 전부 ----
    table = band_rate_table(h_flat)
    rmin_a = np.empty(len(combos)); rmin_f = np.empty(len(combos))
    for a in range(0, len(combos), 20_000):
        c4, inv = placements(seq, combos[a:a + 20_000])
        va, _ = best_assignment(table, c4)
        vf = table[c4, FIXED[None, :]].min(axis=1)
        rmin_a[a:a + 20_000] = va[inv].min(axis=1)
        rmin_f[a:a + 20_000] = vf[inv].min(axis=1)
    print(f"OFDMA done {time.time() - t0:.0f} s", flush=True)
    coarse_o = np.load(config.PROJECT_ROOT / "code" / "ofdma_results.npy", allow_pickle=True).item()

    # ---- MU-MIMO: 상위 3,000 + 무작위 3,000 ----
    coarse_m = np.load(config.PROJECT_ROOT / "code" / "mumimo_results.npy", allow_pickle=True).item()
    top = np.argsort(coarse_m["rmin_bps"])[::-1][:3000]
    rand = RNG.choice(len(combos), 3000, replace=False)
    sel = np.unique(np.r_[top, rand])
    c4, inv = placements(seq, combos[sel])
    worst = zf_rates_bps(h_flat, c4).min(axis=1)[inv]
    rmin_m_sel = worst.min(axis=1)
    print(f"MU-MIMO done {time.time() - t0:.0f} s ({len(c4):,} placements)", flush=True)
    rmin_m = dict(zip(sel.tolist(), rmin_m_sel))
    best_m = max(rmin_m, key=rmin_m.get)
    in_rand = np.isin(sel, rand)

    summary = {
        "fine_map": {"resolution_m": FINE_RES_M, "points": int(ny * nx), "free_points": int(len(meta["free_idx"]))},
        "ofdma_adaptive": {"coarse_best_mbps": round(float(coarse_o["rmin_adapt_bps"].max()) / 1e6, 2),
                           "fine_best_mbps": round(float(rmin_a.max()) / 1e6, 2),
                           "fine_best_paths": dict(zip(AMRS, map(int, combos[np.argmax(rmin_a)]))),
                           "fine_value_of_coarse_best_mbps": round(float(rmin_a[np.argmax(coarse_o["rmin_adapt_bps"])]) / 1e6, 2),
                           "fine_median_mbps": round(float(np.median(rmin_a)) / 1e6, 2),
                           "rank_corr_coarse_vs_fine": round(spearman(coarse_o["rmin_adapt_bps"], rmin_a), 3)},
        "ofdma_fixed": {"coarse_best_mbps": round(float(coarse_o["rmin_fixed_bps"].max()) / 1e6, 2),
                        "fine_best_mbps": round(float(rmin_f.max()) / 1e6, 2)},
        "mumimo": {"coarse_best_mbps": round(float(coarse_m["rmin_bps"].max()) / 1e6, 2),
                   "fine_best_mbps_among_evaluated": round(float(rmin_m[best_m]) / 1e6, 2),
                   "fine_best_paths": dict(zip(AMRS, map(int, combos[best_m]))),
                   "fine_value_of_coarse_best_mbps": round(float(rmin_m[int(top[0])]) / 1e6, 2),
                   "best_is_from_coarse_top3000": bool(best_m in set(top.tolist())),
                   "rank_corr_coarse_vs_fine_random3000": round(spearman(coarse_m["rmin_bps"][sel[in_rand]], rmin_m_sel[in_rand]), 3),
                   "fine_median_random3000_mbps": round(float(np.median(rmin_m_sel[in_rand])) / 1e6, 2)},
        "runtime_s": round(time.time() - t0, 1),
    }
    np.save(config.PROJECT_ROOT / "code" / "fine_eval_results.npy",
            {"combos": combos, "ofdma_adapt": rmin_a, "ofdma_fixed": rmin_f, "mumimo_sel": sel, "mumimo_rmin": rmin_m_sel},
            allow_pickle=True)
    (config.FIGURES_DIR / "07_fine_vs_coarse_summary.json").write_text(json.dumps(summary, indent=2))
    plot_maps(H, meta)
    plot_compare(coarse_o, rmin_a, coarse_m, sel, rmin_m_sel, in_rand)
    print(json.dumps(summary, indent=2))


def plot_maps(H, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from fingerprint import load_or_build_fingerprint, snr_grid_db
    fp = load_or_build_fingerprint(verbose=False)
    coarse = snr_grid_db(fp["H"], "mrc").mean(axis=2); coarse[fp["occupied"]] = np.nan
    fine = 10 * np.log10(np.maximum(SNR_SCALE * np.sum(np.abs(H) ** 2, axis=2), 1e-30)).mean(axis=2)
    fine[meta["occupied"]] = np.nan
    fine_sc = 10 * np.log10(np.maximum(SNR_SCALE * np.sum(np.abs(H) ** 2, axis=2)[..., 0], 1e-30))  # 부반송파 0
    fine_sc[meta["occupied"]] = np.nan
    ext = (0, config.ROOM_X_M, 0, config.ROOM_Y_M)
    vmin, vmax = np.nanpercentile(fine, 1), np.nanmax(fine)
    fig, axes = plt.subplots(2, 2, figsize=(17, 9.5))
    for ax, data, title in ((axes[0, 0], coarse, "50 cm map (512 points)"), (axes[0, 1], fine, "10 cm map (12,800 points)")):
        im = ax.imshow(data, origin="lower", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"MRC SNR, mean over 64 subcarriers: {title}")
        ax.plot(*config.BS_CENTER_M[:2], "w^", mec="k", ms=9)
        ax.add_patch(Rectangle((5, 1), 1, 1, fill=False, ec="r", lw=1.5))
        fig.colorbar(im, ax=ax, shrink=0.8, label="dB")
    # 1 m x 1 m 확대 (x 5~6, y 1~2)
    sl = (slice(10, 20), slice(50, 60))
    ax = axes[1, 0]
    im = ax.imshow(fine_sc[sl], origin="lower", extent=(5, 6, 1, 2), cmap="magma")
    fig.colorbar(im, ax=ax, label="dB")
    ax.set_title("zoom x 5-6 m, y 1-2 m (red box): 10 cm map, single subcarrier\n"
                 f"spread {np.nanmax(fine_sc[sl]) - np.nanmin(fine_sc[sl]):.1f} dB inside one 50 cm cell area")
    ax = axes[1, 1]
    cy = 15  # y = 1.55 m 한 줄
    xs = meta["x_m"]
    ax.plot(xs, fine_sc[cy], lw=0.9, label="10 cm map, subcarrier 0")
    ax.plot(xs, fine[cy], lw=1.6, label="10 cm map, mean over 64 subcarriers")
    xc = np.arange(0.25, config.ROOM_X_M, 0.5)
    ax.step(xc, coarse[3], where="mid", color="k", lw=1.4, label="50 cm map (used before), mean")
    ax.set_xlabel("x [m] (y = 1.55 m line)"); ax.set_ylabel("MRC SNR [dB]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("cut along one line: small-scale fading inside 50 cm cells")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_fine_map_snr.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_compare(coarse_o, rmin_a, coarse_m, sel, rmin_m_sel, in_rand):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.scatter(coarse_o["rmin_adapt_bps"] / 1e6, rmin_a / 1e6, s=2, alpha=0.2)
    lim = [min(rmin_a.min(), coarse_o["rmin_adapt_bps"].min()) / 1e6, max(rmin_a.max(), coarse_o["rmin_adapt_bps"].max()) / 1e6]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("R_min with 50 cm map [Mbps]"); ax.set_ylabel("R_min with 10 cm map [Mbps]")
    ax.set_title(f"OFDMA (adaptive bands), all {len(rmin_a):,} combos"); ax.grid(alpha=0.3)
    ax = axes[1]
    cm = coarse_m["rmin_bps"][sel] / 1e6
    ax.scatter(cm[~in_rand], rmin_m_sel[~in_rand] / 1e6, s=4, alpha=0.4, label="top 3,000 by 50 cm map")
    ax.scatter(cm[in_rand], rmin_m_sel[in_rand] / 1e6, s=4, alpha=0.4, label="random 3,000")
    lim = [min(cm.min(), rmin_m_sel.min() / 1e6), max(cm.max(), rmin_m_sel.max() / 1e6)]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("R_min with 50 cm map [Mbps]"); ax.set_ylabel("R_min with 10 cm map [Mbps]")
    ax.set_title("MU-MIMO (4x4 ZF)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_fine_vs_coarse.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
