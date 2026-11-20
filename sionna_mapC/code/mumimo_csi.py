"""MU-MIMO만 사용 (4대가 부반송파 64개 전체 공유, BS 4x4 ZF): CSI가 완벽할 때와 오차가 있을 때.

이 맵의 10 cm SNR 맵, 후보 경로(path_candidates.npy), 충돌 없는 조합(feasible_combos.npy)만 쓴다.

모델 (업링크, 부반송파 n마다, y = H x + z)
* 완벽한 CSI: W = H⁻¹,  SINR_k = P / (σ² [(HᴴH)⁻¹]_kk)          (이전 맵 mumimo.py와 같음)
* CSI 오차:  BS가 아는 채널 Ĥ = H − E,  ZF는 Ĥ로 만든다 (W = Ĥ⁻¹).
    E의 원소는 서로 독립인 복소 가우시안, AMR j 열의 분산 σ_e,j² = NMSE · ‖h_j[n]‖² / N_ant
    (NMSE = E‖e_j‖² / ‖h_j‖², 각 AMR 채널 전력에 대한 상대 오차).
    W·H = I + W·E 이므로 다른 AMR 신호가 간섭으로 남는다. 그 평균 전력을 잡음처럼 취급하면
        SINR_k = P / ((σ² + P Σ_j σ_e,j²) · [(ĤᴴĤ)⁻¹]_kk)
    (MMSE 채널 추정 + ZF에서 널리 쓰는 하한식). 여기서는 [(ĤᴴĤ)⁻¹]_kk ≈ [(HᴴH)⁻¹]_kk 로 근사해서
    난수 없이 결정적으로 계산한다 (NMSE가 작으면 거의 같고, 같은 배치에는 항상 같은 값이 나온다).
* rate_k = Δf Σ_n log₂(1 + SINR_k[n]),  R_min(조합) = 이동 전체 동안 4대 중 가장 낮은 순간 rate의 최솟값.
* 조합 선택: R_min 최대, 같으면 '순간 최저 rate의 시간 평균'이 큰 것 (이전 맵과 같은 규칙).

출력 (완벽한 CSI와 CSI 오차를 따로 저장)
* code/mumimo_perfect_results.npy, figures/04_mumimo_perfect.png, figures/04_mumimo_perfect_summary.json
* code/mumimo_csi_results.npy,     figures/05_mumimo_csi.png,     figures/05_mumimo_csi_summary.json
"""

from __future__ import annotations

import json
import time

import numpy as np

import config
import mumimo
from collision import AMRS
from fine_eval import fine_sequences, load_fine, placements
from fine_map import FINE_RES_M
from path_planner import load_path_sets

SNR_SCALE = config.tx_power_w() / config.noise_power_w()
NMSE_DB = np.array([-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0])
CHUNK = 1_000          # 조합 단위
ZF_CHUNK = 40_000      # 배치 단위


def zf_terms(h_flat: np.ndarray, cells4: np.ndarray):
    """배치마다 [(HᴴH)⁻¹]_kk (u, 64, 4)와 Σ_j ‖h_j‖²/N_ant (u, 64)."""
    gd = np.empty((len(cells4), config.N_SC, 4))
    pw = np.empty((len(cells4), config.N_SC))
    for a in range(0, len(cells4), ZF_CHUNK):
        c = cells4[a:a + ZF_CHUNK]
        H = np.transpose(h_flat[c], (0, 3, 2, 1))                  # (u, 64, ant, user)
        G = np.conj(np.swapaxes(H, -1, -2)) @ H
        gd[a:a + ZF_CHUNK] = np.maximum(np.real(np.diagonal(np.linalg.inv(G), axis1=-2, axis2=-1)), 1e-300)
        pw[a:a + ZF_CHUNK] = np.real(np.einsum("bnak,bnak->bn", H, np.conj(H))) / config.N_ANT_BS
    return gd, pw


def rates_bps(gd: np.ndarray, pw: np.ndarray, nmse_lin: np.ndarray) -> np.ndarray:
    """(L, u, 4) rate. nmse_lin[0] = 0 이면 완벽한 CSI."""
    out = np.empty((len(nmse_lin), gd.shape[0], 4))
    for i, e in enumerate(nmse_lin):
        sinr = SNR_SCALE / ((1.0 + SNR_SCALE * e * pw)[..., None] * gd)
        out[i] = config.DELTA_F_HZ * np.log2(1.0 + sinr).sum(axis=1)
    return out


def main():
    t0 = time.time()
    _, _, h_flat, remap, ny, nx = load_fine()
    paths = load_path_sets()
    combos = np.load(config.PROJECT_ROOT / "code" / "feasible_combos.npy", allow_pickle=True).item()["combos"].astype(int)
    t_end = max(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                for amr in AMRS for p in paths[amr + "_xy"]) / config.AMR_SPEED_MPS
    t = np.arange(0.0, t_end + config.SIM_DT_S, config.SIM_DT_S)
    seq = fine_sequences(paths, t, nx, ny, remap)

    levels = np.r_[0.0, 10 ** (NMSE_DB / 10)]                       # 0번 = 완벽한 CSI
    n, L = len(combos), len(levels)
    rmin, rmean = np.empty((L, n)), np.empty((L, n))
    for a in range(0, n, CHUNK):
        c4, inv = placements(seq, combos[a:a + CHUNK])
        gd, pw = zf_terms(h_flat, c4)
        inst_min = rates_bps(gd, pw, levels).min(axis=2)[:, inv]     # (L, chunk, T)
        rmin[:, a:a + CHUNK], rmean[:, a:a + CHUNK] = inst_min.min(axis=2), inst_min.mean(axis=2)
        done = min(a + CHUNK, n)
        el = time.time() - t0
        if (a // CHUNK) % 10 == 0 or done == n:
            print(f"{done:,}/{n:,} combos  {el / 60:.1f} min, ~{el / done * (n - done) / 60:.1f} min left", flush=True)

    lengths = {amr: np.array([np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum()
                              for p in paths[amr + "_xy"]]) for amr in AMRS}
    makespan = np.max(np.stack([lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS)]), axis=0) / config.AMR_SPEED_MPS
    total = sum(lengths[amr][combos[:, k]] for k, amr in enumerate(AMRS))
    i_ref = int(np.argmin(makespan * 1000 + total))
    best = [int(np.lexsort((rmean[i], rmin[i]))[-1]) for i in range(L)]
    mb = lambda v: round(float(v) / 1e6, 2)

    def entry(i):
        b = best[i]
        return {"best_paths": dict(zip(AMRS, map(int, combos[b]))), "rmin_mbps": mb(rmin[i, b]),
                "combos_tied_at_best": int(np.isclose(rmin[i], rmin[i].max(), rtol=0, atol=1e3).sum()),
                "mean_of_instant_min_mbps": mb(rmean[i, b]), "makespan_s": round(float(makespan[b]), 1),
                "median_over_combos_mbps": mb(np.median(rmin[i])),
                "fastest_combo_rmin_mbps": mb(rmin[i, i_ref]),
                "perfect_csi_best_combo_rmin_mbps": mb(rmin[i, best[0]])}

    common = {"map_folder": config.PROJECT_ROOT.name, "channel": "10 cm Sionna RT map (3 Gaussian impairments)",
              "feasible_combos": n, "fastest_combo": dict(zip(AMRS, map(int, combos[i_ref]))),
              "fastest_makespan_s": round(float(makespan[i_ref]), 1)}

    # ---- 완벽한 CSI ----
    lookup = dict(zip(map(tuple, combos.tolist()), rmin[0]))
    res = {"t": t, "seq": seq, "paths": paths, "h_flat": h_flat, "rmin_of": lambda c: lookup[tuple(int(v) for v in c)]}
    mumimo.plot(res, combos[best[0]], combos[i_ref], config.FIGURES_DIR / "04_mumimo_perfect.png",
                nx=nx, res_m=FINE_RES_M, tag="10 cm map, perfect CSI")
    tr = mumimo.trace_combo(res, combos[best[0]])
    s_perf = {**common, "csi": "perfect", **entry(0),
              "mean_rate_per_amr_mbps": dict(zip(AMRS, np.round(tr["zf"].mean(axis=0) / 1e6, 2).tolist())),
              "runtime_min_both": None}
    np.save(config.PROJECT_ROOT / "code" / "mumimo_perfect_results.npy",
            {"combos": combos, "rmin": rmin[0], "mean_inst_min": rmean[0], "t": t}, allow_pickle=True)

    # ---- CSI 오차 ----
    s_csi = {**common, "csi_model": "H_hat = H - E, E ~ CN(0, NMSE*|h_j|^2/N_ant) per element; "
                                    "SINR_k = P / ((s2 + P*sum_j s_ej^2) [(H^H H)^-1]_kk)",
             "levels": {f"{d:g} dB": entry(i + 1) for i, d in enumerate(NMSE_DB)}}
    np.save(config.PROJECT_ROOT / "code" / "mumimo_csi_results.npy",
            {"combos": combos, "nmse_db": NMSE_DB, "rmin": rmin[1:], "mean_inst_min": rmean[1:], "t": t},
            allow_pickle=True)
    plot_csi(t, seq, h_flat, combos, rmin, best, i_ref, levels)

    rt = round((time.time() - t0) / 60, 1)
    s_perf["runtime_min_both"] = s_csi["runtime_min_both"] = rt
    (config.FIGURES_DIR / "04_mumimo_perfect_summary.json").write_text(json.dumps(s_perf, indent=2))
    (config.FIGURES_DIR / "05_mumimo_csi_summary.json").write_text(json.dumps(s_csi, indent=2))
    print(json.dumps(s_perf, indent=1)); print(json.dumps(s_csi, indent=1))


def plot_csi(t, seq, h_flat, combos, rmin, best, i_ref, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5.8))
    x = NMSE_DB
    ax.plot(x, rmin[1:, [best[i + 1] for i in range(len(x))]].diagonal() / 1e6, "o-", lw=2,
            label="best path combo at each NMSE (re-chosen)")
    ax.plot(x, rmin[1:, best[0]] / 1e6, "s--", label="combo that is best with perfect CSI")
    ax.plot(x, rmin[1:, i_ref] / 1e6, "^:", label="fastest collision-free combo")
    ax.plot(x, np.median(rmin[1:], axis=1) / 1e6, "x-.", color="gray", label="median over all feasible combos")
    ax.axhline(rmin[0, best[0]] / 1e6, color="k", lw=0.8, ls="--", label=f"perfect CSI best ({rmin[0, best[0]] / 1e6:.1f})")
    ax.set_xlabel("CSI NMSE [dB]"); ax.set_ylabel("R_min [Mbps]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("MU-MIMO (4x4 ZF) R_min vs CSI error")

    cells = np.stack([seq[amr][combos[best[0]][k]] for k, amr in enumerate(AMRS)], axis=1)
    gd, pw = zf_terms(h_flat, cells)
    r = rates_bps(gd, pw, levels).min(axis=2)                       # (L, T)
    cmap = plt.get_cmap("viridis")
    ax2.plot(t, r[0] / 1e6, "k", lw=2, label="perfect CSI")
    for i, d in enumerate(NMSE_DB):
        ax2.plot(t, r[i + 1] / 1e6, color=cmap(i / (len(x) - 1)), lw=1.2, label=f"NMSE {d:g} dB")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("min over 4 AMRs of instant rate [Mbps]"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_title(f"Instant min rate over time, paths {dict(zip(AMRS, map(int, combos[best[0]])))}")
    fig.suptitle(f"{config.PROJECT_ROOT.name}: MU-MIMO with imperfect CSI (10 cm map)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_mumimo_csi.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
