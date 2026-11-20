"""Long-term + short-term 채널 시뮬레이션 (Sionna RT, CPU) 실행 스크립트.

    python st_main.py              # 전체 (약 수 분)
    python st_main.py --quick      # 실현 수를 줄인 빠른 확인

먼저 main.py 와 같은 방식으로 long-term(과제 모델) 계획을 세우고, 같은 경로 쌍을
  (1) long-term 격자 모델,  (2) 실제 위치 Sionna RT (결정적 short-term),
  (3) RT + 가우시안 short-term (Monte-Carlo)
에서 평가한다.  결과: ../figures/st_*.png, st_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from main import setup_sionna


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = config.FIGURES_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.name}")
    return p


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(config.RANDOM_SEED)
    setup_sionna()

    import short_term as st
    import visualizer as viz
    from fingerprint import load_or_build
    from path_planner import build_path_sets
    from schemes import MAIN5, SCHEMES, run_ablation
    from simulator import avg_snr_map_db, bilinear

    n_real = 5 if args.quick else config.ST_N_REALIZATIONS
    results: dict = {"config": {k: (v if not isinstance(v, float) or np.isfinite(v) else str(v))
                                for k, v in vars(config).items() if k.startswith("ST_")}}
    results["config"].pop("ST_DIR", None)

    # ------------------------------------------------------------------ long-term plan (과제 모델)
    data = load_or_build()
    H = data["H"]
    grid, paths = build_path_sets()
    res, _ = run_ablation(grid, paths, H)
    print("[long-term] 8 schemes planned on the grid fingerprint")

    # ------------------------------------------------------------------ 1. long-term map vs short-term at grid
    print("\n[1] Sionna RT on the 720 grid cells → long-term (incoherent) vs coherent map")
    X, Y = np.meshgrid(config.GRID_X_M, config.GRID_Y_M)
    gtr = st.rt_tracks(np.c_[X.ravel(), Y.ravel()], verbose=False)
    lt_siso = db(np.abs(gtr.a[:, 0, :]) ** 2 @ np.ones(gtr.a.shape[-1]) * config.P_REF_W / config.N0_W).reshape(X.shape)
    coh_siso = avg_snr_map_db(H, "siso")
    st_grid = coh_siso - lt_siso
    results["grid_short_term_db"] = {"mean": float(st_grid.mean()), "std": float(st_grid.std()),
                                     "min": float(st_grid.min()), "max": float(st_grid.max())}
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for ax, m, title, cmap, lim in (
            (axes[0], coh_siso, "Project fingerprint (coherent sum on 0.5 m grid) — SISO SNR", "viridis", None),
            (axes[1], lt_siso, "Long-term component: local mean Σ|a_i|² (incoherent) — SISO SNR", "viridis", None),
            (axes[2], st_grid, "Short-term (multipath) part frozen into the grid: coherent − long-term [dB]", "RdBu_r", 15)):
        if lim:
            viz.heat(ax, m, -lim, lim, cmap=cmap, label="dB")
        else:
            viz.heat(ax, m, np.percentile(coh_siso, 1), coh_siso.max())
        viz.draw_scene(ax, labels=False)
        ax.set_title(title, fontsize=10)
    _save(fig, "st_01_long_vs_short_map.png")

    # ------------------------------------------------------------------ 2. RT at the exact AMR positions
    print("\n[2] Sionna RT at every AMR position of all 8 schemes (0.1 s = 5 cm steps)")
    pts, index = [], {}
    for name, *_ in SCHEMES:
        for k, pos in (("AMR1", res[name].sim.pos1), ("AMR2", res[name].sim.pos2)):
            index[(name, k)] = (len(pts), len(pts) + len(pos))
            pts.extend(pos.tolist())
    pts = np.array(pts)
    uniq, inv = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    inv = inv.ravel()
    t0 = time.time()
    utr = st.rt_tracks(uniq)
    print(f"  {len(uniq)} unique positions traced in {time.time() - t0:.1f} s")
    track = {key: st.take(utr, inv[s:e]) for key, (s, e) in index.items()}

    # ------------------------------------------------------------------ 3. decomposition along JOINT AMR1
    print("\n[3] long-term / short-term decomposition along the JOINT AMR1 path")
    j = res["JOINT"].sim
    tr = track[("JOINT", "AMR1")]
    lt = db(tr.power_lt * config.P_REF_W / config.N0_W)                                  # long-term
    rt_snr = st.snr_db(st.synthesize(tr, config.SIM_DT_S, False), "siso", 0).mean(1)    # RT exact (평균 SNR dB)
    rt_lin = db((10 ** (st.snr_db(st.synthesize(tr, config.SIM_DT_S, False), "siso", 0) / 10)).mean(1))
    ga_lin = db((10 ** (st.snr_db(st.synthesize(tr, config.SIM_DT_S, True), "siso", 0) / 10)).mean(1))
    grid_lin = bilinear(coh_siso, j.pos1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(j.t, grid_lin, color="tab:gray", lw=2, label="project model: grid fingerprint + bilinear")
    ax.plot(j.t, lt, color="k", lw=2, label="long-term: local mean Σ|a_i|² (Sionna RT)")
    ax.plot(j.t, rt_lin, color="tab:blue", lw=1, label="long + short (multipath, RT at exact position)")
    ax.plot(j.t, ga_lin, color="tab:red", lw=.8, alpha=.8, label="long + short (multipath + Gaussian), 1 realisation")
    ax.set_ylabel("SISO SNR (avg over 64 sc) [dB]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("JOINT AMR1: long-term vs short-term components along the trajectory")
    axes[1].plot(j.t, rt_lin - lt, color="tab:blue", lw=1, label="short-term (multipath) = RT − long-term")
    axes[1].plot(j.t, ga_lin - lt, color="tab:red", lw=.8, alpha=.8, label="short-term (multipath + Gaussian)")
    axes[1].axhline(0, color="k", lw=.8)
    axes[1].set(xlabel="time [s]", ylabel="short-term [dB]"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    _save(fig, "st_02_decomposition_along_path.png")
    results["joint_amr1_short_term_db"] = {"multipath_std": float(np.std(rt_lin - lt)),
                                           "multipath_plus_gauss_std": float(np.std(ga_lin - lt)),
                                           "grid_vs_rt_exact_rms_db": float(np.sqrt(np.mean((grid_lin - rt_lin) ** 2)))}
    del rt_snr

    # ------------------------------------------------------------------ 4. fine-resolution fading (λ scale) + statistics
    print("\n[4] fine segment: Sionna RT every 2 mm over 2 m (λ/25 sampling)")
    xy = j.pos1
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s0 = 2.0
    ss = np.arange(s0, s0 + config.ST_FINE_SEGMENT_M, config.ST_FINE_STEP_M)
    fine_xy = np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])]
    ftr = st.rt_tracks(fine_xy, verbose=False)
    dt_fine = config.ST_FINE_STEP_M / config.AMR_SPEED_MPS["AMR1"]
    Hf = st.synthesize(ftr, dt_fine, False)
    Hg = st.synthesize(ftr, dt_fine, True)
    k0 = config.N_SC // 2
    p_rt = np.abs(Hf[:, 0, k0]) ** 2
    p_ga = np.abs(Hg[:, 0, k0]) ** 2
    p_lt = (np.abs(ftr.a[:, 0, :]) ** 2).sum(-1)
    # 통계용: 모든 실현·부반송파·안테나의 정규화 전력 |h|^2 / 국소평균
    norm_rt, norm_ga = [], []
    for _ in range(10):
        Hg_ = st.synthesize(ftr, dt_fine, True)
        norm_ga.append((np.abs(Hg_) ** 2 / ftr.power_lt[:, None, None]).ravel())
    norm_rt = (np.abs(Hf) ** 2 / ftr.power_lt[:, None, None]).ravel()
    norm_ga = np.concatenate(norm_ga)
    lag = np.arange(0, 100)
    env = np.abs(Hf[:, 0, k0]) - np.abs(Hf[:, 0, k0]).mean()
    acf = np.array([np.mean(env[:len(env) - L] * env[L:]) for L in lag]) / np.mean(env ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ss, db(p_lt * config.P_REF_W / config.N0_W), "k", lw=2, label="long-term")
    axes[0].plot(ss, db(p_rt * config.P_REF_W / config.N0_W), color="tab:blue", label="RT multipath")
    axes[0].plot(ss, db(p_ga * config.P_REF_W / config.N0_W), color="tab:red", lw=.8, label="+ Gaussian")
    axes[0].set(xlabel="distance along path [m]", ylabel="SNR (ant 0, k=32) [dB]", title="λ-scale fading (2 mm steps)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    xs = np.linspace(-30, 10, 300)
    for v, c, lab in ((norm_rt, "tab:blue", "RT multipath"), (norm_ga, "tab:red", "RT + Gaussian")):
        vv = np.sort(db(v))
        axes[1].plot(vv, np.arange(1, len(vv) + 1) / len(vv), color=c, label=lab)
    axes[1].plot(xs, 1 - np.exp(-10 ** (xs / 10)), "k--", lw=1, label="Rayleigh (reference)")
    axes[1].set(yscale="log", ylim=(1e-3, 1), xlim=(-30, 10), xlabel="|h|² / local mean [dB]", ylabel="CDF",
                title="short-term power distribution")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3, which="both")
    axes[2].plot(lag * config.ST_FINE_STEP_M / config.LAMBDA_C_M, acf, color="tab:blue")
    axes[2].axhline(0.5, color="gray", ls=":")
    axes[2].set(xlabel="displacement [λ]", ylabel="envelope autocorrelation", title="spatial coherence of short-term fading")
    axes[2].grid(alpha=.3)
    _save(fig, "st_03_fine_fading_statistics.png")
    coh = lag[np.argmax(acf < 0.5)] * config.ST_FINE_STEP_M if np.any(acf < 0.5) else np.nan
    results["fine_segment"] = {"coherence_distance_m_at_acf_0.5": float(coh),
                               "coherence_distance_over_lambda": float(coh / config.LAMBDA_C_M),
                               "short_term_db_std_rt": float(np.std(db(norm_rt))),
                               "short_term_db_std_rt_gauss": float(np.std(db(norm_ga))),
                               "p_below_minus10db_rt": float(np.mean(norm_rt < 0.1)),
                               "p_below_minus10db_rt_gauss": float(np.mean(norm_ga < 0.1))}

    # ------------------------------------------------------------------ 5. Gaussian short-term only (AMR standing still)
    print("\n[5] AMR standing still at its goal: only the Gaussian (air flow / insects) part varies")
    T_still = 200
    still = st.take(utr, np.full(T_still, inv[index[("JOINT", "AMR1")][1] - 1]))
    t_still = np.arange(T_still) * config.SIM_DT_S
    fig, ax = plt.subplots(figsize=(11, 3.8))
    base = db(np.abs(st.synthesize(still, config.SIM_DT_S, False)[:, 0, k0]) ** 2 * config.P_REF_W / config.N0_W)
    ax.plot(t_still, base, "k", lw=2, label="RT (deterministic, AMR at rest)")
    for i, c in enumerate(("tab:red", "tab:orange", "tab:purple")):
        Hs = st.synthesize(still, config.SIM_DT_S, True)
        ax.plot(t_still, db(np.abs(Hs[:, 0, k0]) ** 2 * config.P_REF_W / config.N0_W), color=c, lw=.9,
                label=f"+ Gaussian short-term, realisation {i + 1}")
    ax.set(xlabel="time [s]", ylabel="SNR (ant 0, k=32) [dB]",
           title=f"Gaussian short-term only: σ_d={config.ST_PATH_LEN_JITTER_STD_M*1e3:.0f} mm, σ_A={config.ST_PATH_AMP_JITTER_STD}, "
                 f"K={config.ST_DIFFUSE_K_DB:.0f} dB, T_c={config.ST_PATH_COHERENCE_S}/{config.ST_DIFFUSE_COHERENCE_S} s")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, "st_04_gaussian_only_at_rest.png")

    # ------------------------------------------------------------------ 6. all 8 schemes under long + short term
    print(f"\n[6] evaluating 8 schemes, {n_real} Gaussian realisations each")
    ev = {}
    for name, *_ in SCHEMES:
        ev[name] = st.evaluate(res[name], track[(name, "AMR1")], track[(name, "AMR2")], n_real)
        sm = ev[name]["summary"]
        print(f"  {name:<11s} Rmin: grid {sm['long_term_grid']['rmin_mbps']:6.2f} | RT {sm['rt_exact']['rmin_mbps']:6.2f}"
              f" | RT+G offline-k* {sm['lt_plus_st_offline_kstar']['rmin_mbps']:6.2f}"
              f" (p10 {sm['lt_plus_st_offline_kstar']['rmin_p10_mbps']:6.2f})"
              f" | adaptive {sm['lt_plus_st_adaptive']['rmin_mbps']:6.2f}"
              f" | outage {sm['lt_plus_st_offline_kstar']['outage_frac']*100:5.1f}%")
    results["schemes"] = {n: ev[n]["summary"] for n in ev}

    names = [s[0] for s in SCHEMES]
    keys = (("long_term_grid", "long-term only (project model)", "tab:gray"),
            ("rt_exact", "RT exact position (multipath)", "tab:blue"),
            ("lt_plus_st_offline_kstar", "long + short (Gaussian), offline k*", "tab:red"),
            ("lt_plus_st_adaptive", "long + short (Gaussian), adaptive k*", "tab:green"))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    w = 0.2
    xpos = np.arange(len(names))
    for i, (k, lab, c) in enumerate(keys):
        v = [ev[n]["summary"][k]["rmin_mbps"] for n in names]
        b = axes[0].bar(xpos + (i - 1.5) * w, v, w, color=c, label=lab, edgecolor="k", lw=.5)
        if k.startswith("lt_plus"):
            p10 = [ev[n]["summary"][k]["rmin_p10_mbps"] for n in names]
            axes[0].errorbar(xpos + (i - 1.5) * w, v, yerr=[np.array(v) - np.array(p10), np.zeros(len(v))],
                             fmt="none", ecolor="k", capsize=2, lw=.8)
        axes[0].bar_label(b, fmt="%.0f", fontsize=6)
        axes[1].bar(xpos + (i - 1.5) * w, [ev[n]["summary"][k]["outage_frac"] * 100 for n in names], w, color=c,
                    edgecolor="k", lw=.5)
    axes[0].axhline(config.R_QOS_MBPS, color="red", ls="--", lw=1)
    axes[0].set_ylabel("R_min [Mbps]  (mean over realisations; bar = 10th pct.)"); axes[0].legend(fontsize=8)
    axes[1].set_ylabel(f"outage: % of time R_min(t) < {config.R_QOS_MBPS:.0f} Mbps")
    axes[1].set_xticks(xpos, names, rotation=15)
    for a in axes:
        a.grid(axis="y", alpha=.3)
    fig.suptitle(f"8 schemes under long-term only vs long-term + short-term (Sionna RT, {n_real} realisations)")
    _save(fig, "st_05_schemes_long_vs_short.png")

    fig, axes = plt.subplots(len(MAIN5), 1, figsize=(11, 2.3 * len(MAIN5)))
    for ax, n in zip(axes, MAIN5):
        sim, e = res[n].sim, ev[n]
        ax.plot(sim.t, e["lt_trace"] / 1e6, color="tab:gray", lw=2, label="long-term only")
        ax.plot(sim.t, e["rt_trace"] / 1e6, color="tab:blue", lw=1, label="RT multipath")
        m, sd = e["st_off"].mean(0) / 1e6, e["st_off"].std(0) / 1e6
        ax.plot(sim.t, m, color="tab:red", lw=1, label="long + short (Gaussian) mean")
        ax.fill_between(sim.t, m - sd, m + sd, color="tab:red", alpha=.2, label="±1 std")
        ax.axhline(config.R_QOS_MBPS, color="red", ls="--", lw=.8)
        ax.set_ylabel(f"{n}\n[Mbps]"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, ncol=4, loc="lower left")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("R_min(t): long-term only vs long-term + short-term (offline k* reused)")
    fig.tight_layout()
    _save(fig, "st_06_throughput_traces.png")

    # ------------------------------------------------------------------ 7. sensitivity to the Gaussian (diffuse) strength
    print("\n[7] Rician K sweep (diffuse Gaussian strength)")
    sweep = {}
    nk = max(3, n_real // 3)
    for n in ("NAIVE", "+MRC", "+OPTBW", "JOINT"):
        sweep[n] = []
        for kdb in config.ST_K_SWEEP_DB:
            e = st.evaluate(res[n], track[(n, "AMR1")], track[(n, "AMR2")], nk, k_db=kdb)
            sweep[n].append(e["summary"]["lt_plus_st_offline_kstar"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for n in sweep:
        axes[0].plot(config.ST_K_SWEEP_DB, [v["rmin_mbps"] for v in sweep[n]], "o-", color=viz.COLORS[n], label=n)
        axes[1].plot(config.ST_K_SWEEP_DB, [v["outage_frac"] * 100 for v in sweep[n]], "o-", color=viz.COLORS[n], label=n)
    for a in axes:
        a.invert_xaxis(); a.grid(alpha=.3); a.set_xlabel("Rician K [dB]  (← weaker Gaussian   stronger →)")
    axes[0].axhline(config.R_QOS_MBPS, color="red", ls="--")
    axes[0].set_ylabel("E[R_min] [Mbps]"); axes[1].set_ylabel("outage [% of time]")
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensitivity to the Gaussian short-term (diffuse) component")
    _save(fig, "st_07_k_sweep.png")
    results["k_sweep"] = {n: [{"K_db": k, **v} for k, v in zip(config.ST_K_SWEEP_DB, sweep[n])] for n in sweep}

    with open(config.FIGURES_DIR / "st_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=float)
    print(f"\nDone in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
