"""모든 그림(PNG)과 애니메이션(MP4, ffmpeg 없으면 GIF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

import config
from simulator import avg_snr_map_db, bilinear

FIG = config.FIGURES_DIR
COLORS = {"NAIVE": "tab:gray", "+MRC": "tab:green", "+OPTBW": "tab:purple",
          "+SMARTPATH": "tab:orange", "JOINT": "tab:red", "MRC OPTBW": "tab:olive",
          "MRC SMART": "tab:cyan", "OPT SMART": "tab:brown"}
EXTENT = (config.GRID_X_M[0] - 0.25, config.GRID_X_M[-1] + 0.25,
          config.GRID_Y_M[0] - 0.25, config.GRID_Y_M[-1] + 0.25)


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_scene(ax, labels: bool = True, endpoints: bool = True) -> None:
    for ob in config.OBSTACLES:
        ec = "white" if ob.material == "concrete" else "#f4c27a"
        if ob.kind == "box":
            ax.add_patch(Rectangle((ob.cx - ob.dx / 2, ob.cy - ob.dy / 2), ob.dx, ob.dy, fill=False, ec=ec, lw=1.4))
        else:
            ax.add_patch(Circle((ob.cx, ob.cy), ob.r, fill=False, ec=ec, lw=1.4))
        if labels:
            ax.text(ob.cx, ob.cy + (ob.dy / 2 if ob.kind == "box" else ob.r) + 0.15, ob.name,
                    color=ec, fontsize=6, ha="center", va="bottom")
    ax.plot(*config.BS_CENTER_M[:2], marker="*", ms=16, color="red", mec="white", ls="none", label="BS")
    if endpoints:
        for amr, c in (("AMR1", "tab:blue"), ("AMR2", "tab:orange")):
            (sx, sy), (gx, gy) = config.AMR_START_GOAL[amr]
            ax.plot(sx, sy, "s", color=c, mec="k", ms=8)
            ax.plot(gx, gy, "^", color=c, mec="k", ms=9)
    ax.set_xlim(0, config.ROOM_X_M); ax.set_ylim(0, config.ROOM_Y_M)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def heat(ax, values, vmin=None, vmax=None, cmap="viridis", label="SNR [dB]"):
    im = ax.imshow(values, origin="lower", extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=label)
    return im


# ----------------------------------------------------------------------------
# Task 1
# ----------------------------------------------------------------------------
def plot_snr_maps(H, grid, paths) -> list[Path]:
    siso, mimo = avg_snr_map_db(H, "siso"), avg_snr_map_db(H, "mrc")
    vmin, vmax = np.percentile(siso, 1), mimo.max()
    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    for ax, m, title in ((axes[0], siso, "SISO (antenna 0)"), (axes[1], mimo, "MIMO 4-ant MRC")):
        heat(ax, m, vmin, vmax)
        draw_scene(ax)
        for amr, c in (("AMR1", "tab:blue"), ("AMR2", "tab:orange")):
            xy = grid.xy_of(paths[amr][0])
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=2.2, ls="-" if amr == "AMR1" else "--", label=f"{amr} shortest")
        ax.set_title(f"T1.3 average SNR map: {title} (Sionna RT fingerprint)")
        ax.legend(loc="upper right", fontsize=7)
    out = [_save(fig, "02_snr_heatmaps.png")]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    gain = mimo - siso
    heat(ax, gain, cmap="magma", label="MIMO − SISO [dB]")
    draw_scene(ax, labels=False)
    ax.set_title(f"T1.3 Q3: MRC array gain  mean={gain.mean():.2f} dB, std={gain.std():.2f} dB")
    out.append(_save(fig, "02b_mrc_gain_map.png"))
    return out


def plot_along_path(H, grid, paths) -> tuple[Path, list[dict]]:
    siso = avg_snr_map_db(H, "siso")
    xy = grid.xy_of(paths["AMR1"][0])
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    dense = np.linspace(0, s[-1], 300)
    pts = np.c_[np.interp(dense, s, xy[:, 0]), np.interp(dense, s, xy[:, 1])]
    vals = bilinear(siso, pts)
    sample_s = np.linspace(0, s[-1], 6)
    sp = np.c_[np.interp(sample_s, s, xy[:, 0]), np.interp(sample_s, s, xy[:, 1])]
    sv = bilinear(siso, sp)
    rows = [{"s_m": float(a), "x": float(p[0]), "y": float(p[1]), "siso_snr_db": float(v)} for a, p, v in zip(sample_s, sp, sv)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(dense, vals, color="tab:blue", label="SISO SNR along AMR1 shortest")
    ax.plot(sample_s, sv, "ko")
    for a, p, v in zip(sample_s, sp, sv):
        ax.annotate(f"({p[0]:.1f},{p[1]:.1f})\n{v:.1f} dB", (a, v), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.set(xlabel="distance along path [m]", ylabel="SNR [dB]", title="T1.3 Q2: SNR along AMR1 shortest path")
    ax.grid(alpha=.3)
    return _save(fig, "03_along_path_snr.png"), rows


# ----------------------------------------------------------------------------
# Task 2
# ----------------------------------------------------------------------------
def plot_candidates(H, grid, paths, joint_pair) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    heat(ax, avg_snr_map_db(H, "mrc"))
    draw_scene(ax, labels=False)
    for amr, c, k in (("AMR1", "tab:blue", 0), ("AMR2", "tab:orange", 1)):
        for p in paths[amr]:
            xy = grid.xy_of(p)
            ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, alpha=0.25)
        xy = grid.xy_of(paths[amr][joint_pair[k]])
        ax.plot(xy[:, 0], xy[:, 1], color="navy" if amr == "AMR1" else "darkorange", lw=3, label=f"{amr} JOINT")
    ax.imshow(np.ma.masked_where(~grid.blocked, grid.blocked), origin="lower", extent=EXTENT, cmap="gray", alpha=.35)
    ax.set_title("T2.2: 30 candidate paths per AMR (seed 42) on MIMO map, JOINT choice bold")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "04_candidate_paths.png")


def plot_paths_by_scheme(H, grid, paths, res) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    heat(ax, avg_snr_map_db(H, "mrc"))
    draw_scene(ax, labels=False)
    for name, color, lw in (("NAIVE", "lightgray", 4), ("+SMARTPATH", "tab:orange", 3), ("JOINT", "red", 2.2)):
        pair = res[name].pair
        for k, amr in enumerate(("AMR1", "AMR2")):
            xy = grid.xy_of(paths[amr][pair[k]])
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls="-" if k == 0 else "--",
                    label=f"{name if name != 'NAIVE' else 'Shortest (NAIVE/+MRC/+OPTBW)'} {amr}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, ncol=3)
    ax.set_title("T2.3: paths by scheme (solid AMR1, dashed AMR2; ■ start, ▲ goal)")
    return _save(fig, "05_paths_by_scheme.png")


# ----------------------------------------------------------------------------
# Task 3
# ----------------------------------------------------------------------------
def plot_bw_share(res, names) -> Path:
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2 * len(names)), sharex=True)
    for ax, n in zip(axes, names):
        sim = res[n].sim
        share = sim.k_star / config.N_SC
        ax.stackplot(sim.t, share, 1 - share, colors=["tab:blue", "tab:orange"], labels=["AMR1", "AMR2"], alpha=.8)
        ax.set_ylim(0, 1); ax.set_ylabel(f"{n}\nBW share")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("T3.1: BW share of AMR1 / AMR2 over time (k*/N)")
    return _save(fig, "06_bw_allocation.png")


def plot_throughput(res, names) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    lines = []
    for n in names:
        sim = res[n].sim
        ax.plot(sim.t, sim.rmin_t / 1e6, color=COLORS[n], lw=1.6, label=n)
        ax.axvline(sim.T, color=COLORS[n], ls=":", lw=1.3)
        lines.append(f"{n:<11s} Rmin={sim.rmin/1e6:6.1f}  mean={sim.mean/1e6:6.1f}  T={sim.T:5.1f}s")
    ax.axhline(config.R_QOS_MBPS, color="red", ls="--", lw=1.4, label=f"R_QoS = {config.R_QOS_MBPS:.0f} Mbps")
    ax.text(0.01, 0.02, "\n".join(lines), transform=ax.transAxes, family="monospace", fontsize=7.5,
            va="bottom", bbox=dict(fc="white", alpha=.85))
    ax.set(xlabel="time [s]", ylabel="R_min(t) = min(R1, R2) [Mbps]", title="T3.2: worst-AMR throughput vs time")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.3)
    return _save(fig, "01_throughput.png")


def plot_bottleneck(res, names) -> Path:
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.1 * len(names)), sharex=True)
    for ax, n in zip(axes, names):
        sim = res[n].sim
        ax.plot(sim.t, sim.r1 / 1e6, color="tab:blue", label="R1 (AMR1)")
        ax.plot(sim.t, sim.r2 / 1e6, color="tab:orange", label="R2 (AMR2)")
        bott = sim.r2 < sim.r1
        ax.fill_between(sim.t, 0, 1, where=bott, transform=ax.get_xaxis_transform(), color="tab:orange", alpha=.12)
        ax.axhline(config.R_QOS_MBPS, color="red", ls="--", lw=1)
        ax.set_ylabel(f"{n}\n[Mbps]")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time [s]  (shaded = AMR2 is the bottleneck)")
    fig.suptitle("T3.2 Q1: per-AMR rate and bottleneck AMR")
    return _save(fig, "03_bottleneck_per_scheme.png")


def plot_ablation(res, deltas) -> Path:
    from schemes import SCHEMES
    names = [s[0] for s in SCHEMES]
    vals = [[res[n].sim.rmin / 1e6 for n in names], [res[n].sim.mean / 1e6 for n in names], [res[n].sim.T for n in names]]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, v, lab in zip(axes, vals, ("R_min [Mbps]", "mean throughput [Mbps]", "completion time T [s]")):
        b = ax.bar(names, v, color=[COLORS[n] for n in names], edgecolor="k")
        ax.bar_label(b, fmt="%.1f", fontsize=8)
        ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3)
    axes[0].axhline(config.R_QOS_MBPS, color="red", ls="--", lw=1)
    a, d = deltas["add"], deltas["drop"]
    txt = ("            add(+X)   drop(J-X)\n" + "\n".join(f"{k:<10s}{a[k]:+8.2f}  {d[k]:+8.2f}" for k in a)
           + f"\nΣadd = {sum(a.values()):+.2f}\nJOINT−NAIVE = {deltas['joint_minus_naive']:+.2f}  [Mbps]")
    axes[0].text(0.01, 0.97, txt, transform=axes[0].transAxes, family="monospace", fontsize=8, va="top",
                 bbox=dict(fc="white", alpha=.9))
    axes[2].tick_params(axis="x", rotation=20)
    fig.suptitle("T3.3: full 2^3 ablation (Path × BW × Antenna)")
    return _save(fig, "10_ablation_bars.png")


# ----------------------------------------------------------------------------
# Task 4
# ----------------------------------------------------------------------------
def plot_pareto(scores, frontier, joint_pair) -> Path:
    T, R = scores.T.ravel(), scores.rmin.ravel() / 1e6
    ok = R >= config.R_QOS_MBPS
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(T[~ok], R[~ok], c="gray", alpha=.4, s=14, label=f"infeasible ({(~ok).sum()})")
    ax.scatter(T[ok], R[ok], c="tab:blue", alpha=.7, s=14, label=f"feasible ({ok.sum()})")
    if len(frontier):
        ax.plot(T[frontier], R[frontier], "k-", lw=1.8, drawstyle="steps-post", label="Pareto frontier (feasible)")
        ax.plot(T[frontier], R[frontier], "k.", ms=6)
    j = np.ravel_multi_index(joint_pair, scores.T.shape)
    ax.scatter(T[j], R[j], marker="*", s=380, c="yellow", edgecolors="k", zorder=5, label="JOINT choice")
    ax.axhline(config.R_QOS_MBPS, color="red", ls="--", label="R_QoS = 70 Mbps")
    ax.set(xlabel="completion time T [s]", ylabel="R_min [Mbps]", title="T4.1: 900 path pairs (MRC + Opt-BW)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    return _save(fig, "11_pareto.png")


def plot_qos_sweep(rows, name="12_qos_sweep.png", title_extra="") -> Path:
    q = np.array([r["qos_mbps"] for r in rows]); f = np.array([r["feasible"] for r in rows])
    T = np.array([r["T_s"] for r in rows]); R = np.array([r["rmin_mbps"] for r in rows])
    L1 = np.array([r["L1_m"] for r in rows]); L2 = np.array([r["L2_m"] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, y in ((axes[0], T), (axes[1], R)):
        ax.plot(q[f], y[f], "o-", color="green", label="feasible")
        ax.plot(q[~f], y[~f], "x", color="red", ms=10, mew=2, label="infeasible (max-R_min fallback)")
        ax.grid(alpha=.3)
    for qi, ti in zip(q, T):
        axes[0].annotate(f"{ti:.1f}", (qi, ti), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    axes[1].plot([q.min(), q.max()], [q.min(), q.max()], "r--", label="R = R_QoS")
    axes[0].set_ylabel("T [s]"); axes[1].set_ylabel("achieved R_min [Mbps]")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    axes[2].plot(q, L1, "s-", color="tab:blue", label="L1 (AMR1)")
    axes[2].plot(q, L2, "^--", color="tab:orange", label="L2 (AMR2)")
    axes[2].plot(q[~f], L2[~f], "x", color="red", ms=10, mew=2)
    axes[2].set(xlabel="R_QoS [Mbps]", ylabel="path length [m]"); axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)
    fig.suptitle("T4.2: R_QoS sweep (JOINT stack)" + title_extra)
    return _save(fig, name)


# ----------------------------------------------------------------------------
# Task 5
# ----------------------------------------------------------------------------
def plot_online(res, robust, names) -> Path:
    fig, axes = plt.subplots(len(names), 1, figsize=(10.5, 2.2 * len(names)))
    for ax, n in zip(axes, names):
        sim, rb = res[n].sim, robust[n]
        ax.plot(sim.t, sim.rmin_t / 1e6, "k", lw=1.4, label="offline (deterministic)")
        ax.plot(sim.t, rb["mean_t"] / 1e6, color="tab:blue", lw=1.2, label="online mean (Rayleigh)")
        ax.fill_between(sim.t, (rb["mean_t"] - rb["std_t"]) / 1e6, (rb["mean_t"] + rb["std_t"]) / 1e6,
                        color="tab:blue", alpha=.25, label="±1 std")
        ax.set_ylabel(f"{n}\n[Mbps]"); ax.grid(alpha=.3)
        ax.text(.99, .92, f"E[Rmin_on]/Rmin_off = {rb['ratio']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", alpha=.8))
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"T5.3: offline k*(t) replayed on {config.N_REALIZATIONS} Rayleigh realisations (no replanning)")
    fig.tight_layout()
    return _save(fig, "13_offline_vs_online.png")


def mrc_beam(h: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """g(θ) = |w^H a(θ)|^2,  w = h*/||h||,  a_n(θ) = e^{j2π(d/λ) n sinθ}, d = λ/2.

    θ는 배열 broadside(+y) 기준 각이며, 방(top-down) 방위각 φ(+x에서 반시계)와 θ = φ − 90° 관계로
    그린다. 이 규약에서 주엽(main lobe)이 BS→AMR1 방향을 가리킨다.
    """
    theta = phi - np.pi / 2
    n = np.arange(config.N_ANT_BS)
    a = np.exp(1j * np.pi * n[None, :] * np.sin(theta)[:, None])
    w = h.conj() / max(np.linalg.norm(h), 1e-30)
    g = np.abs(a @ w.conj()) ** 2
    return g / g.max()


def animate(res_scheme, H, grid, beam: bool) -> Path:
    sim = res_scheme.sim
    mp = avg_snr_map_db(H, "mrc" if res_scheme.antenna == "mrc" else "siso")
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(3, 3, width_ratios=(2.3, 1, 1), height_ratios=(1.2, .5, 1))
    ax_map = fig.add_subplot(gs[:, 0]); ax_r = fig.add_subplot(gs[0, 1:])
    ax_bw = fig.add_subplot(gs[1, 1:]); ax_txt = fig.add_subplot(gs[2, 1])
    heat(ax_map, mp); draw_scene(ax_map, labels=False)
    ax_map.set_title(f"{res_scheme.name}: AMRs on {res_scheme.antenna.upper()} SNR map")
    tr1, = ax_map.plot([], [], color="cyan", lw=2, alpha=.7)
    tr2, = ax_map.plot([], [], color="orange", lw=2, alpha=.7, ls="--")
    m1, = ax_map.plot([], [], "o", color="cyan", mec="k", ms=10, label="AMR1")
    m2, = ax_map.plot([], [], "o", color="orange", mec="k", ms=10, label="AMR2")
    ax_map.legend(loc="upper right", fontsize=8)
    ax_r.plot(sim.t, sim.rmin_t / 1e6, color="tab:purple", label="R_min(t)")
    ax_r.axhline(config.R_QOS_MBPS, color="red", ls="--", label="QoS")
    cur = ax_r.axvline(0, color="k", ls=":")
    ax_r.set(xlabel="t [s]", ylabel="Mbps"); ax_r.legend(fontsize=7); ax_r.grid(alpha=.3)
    b1 = ax_bw.barh([0], [.5], color="tab:blue", label="AMR1")[0]
    b2 = ax_bw.barh([0], [.5], left=[.5], color="tab:orange", label="AMR2")[0]
    ax_bw.set(xlim=(0, 1), yticks=[], xlabel="subcarrier share  (split at k*)"); ax_bw.legend(fontsize=7, ncol=2, loc="upper center")
    ax_txt.axis("off")
    txt = ax_txt.text(0, 1, "", va="top", family="monospace", fontsize=10)
    phi = np.linspace(0, 2 * np.pi, 361)
    if beam:
        ax_b = fig.add_subplot(gs[2, 2], projection="polar")
        bl, = ax_b.plot([], [], color="tab:red")
        dl, = ax_b.plot([], [], color="k", ls="--", lw=1)
        ax_b.set_title("AMR1 MRC beam |w^H a|^2 (k=32)", fontsize=8); ax_b.set_rticks([.5, 1]); ax_b.tick_params(labelsize=6)
        Hk = H[:, :, :, config.N_SC // 2]
        h_t = bilinear(Hk.real, sim.pos1) + 1j * bilinear(Hk.imag, sim.pos1)      # 지문에서 조회 (재추적 없음)
    stride, fps = 3, 10
    frames = np.r_[np.arange(0, len(sim.t), stride), len(sim.t) - 1]

    def update(f):
        i = int(frames[f])
        tr1.set_data(sim.pos1[:i + 1, 0], sim.pos1[:i + 1, 1]); tr2.set_data(sim.pos2[:i + 1, 0], sim.pos2[:i + 1, 1])
        m1.set_data([sim.pos1[i, 0]], [sim.pos1[i, 1]]); m2.set_data([sim.pos2[i, 0]], [sim.pos2[i, 1]])
        cur.set_xdata([sim.t[i]] * 2)
        s = sim.k_star[i] / config.N_SC
        b1.set_width(s); b2.set_x(s); b2.set_width(1 - s)
        txt.set_text(f"t    = {sim.t[i]:5.1f} s\nR1   = {sim.r1[i]/1e6:6.1f} Mbps\nR2   = {sim.r2[i]/1e6:6.1f} Mbps\n"
                     f"Rmin = {sim.rmin_t[i]/1e6:6.1f} Mbps\nk*   = {sim.k_star[i]:2d}")
        if beam:
            bl.set_data(phi, mrc_beam(h_t[i], phi))
            v = np.r_[sim.pos1[i], config.RX_HEIGHT_M] - config.BS_CENTER_M
            ang = np.arccos(v[0] / np.linalg.norm(v)) * (1 if v[1] >= 0 else -1)
            dl.set_data([ang, ang], [0, 1])
        return []

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100)
    fig.tight_layout()
    stem = "15_joint_animation_mrc_beam" if beam else "14_naive_animation"
    FIG.mkdir(parents=True, exist_ok=True)
    try:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        out = FIG / f"{stem}.mp4"
        ani.save(out, writer=FFMpegWriter(fps=fps, bitrate=1800))
    except Exception as e:                                       # ffmpeg 없음 → GIF fallback
        print(f"  (MP4 실패: {e}) → GIF로 저장")
        out = FIG / f"{stem}.gif"
        ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def render_sionna_scene(H, grid, paths) -> Path | None:
    """Sionna RT 렌더러로 3D 장면 + 첫 셀 경로 그림 (선택, 느릴 수 있음)."""
    try:
        import mitsuba as mi
        import sionna.rt as rt
        from scene_builder import build_scene_files
        _, render_xml = build_scene_files()
        scene = rt.load_scene(str(render_xml), merge_shapes=False)
        scene.frequency = config.FREQ_HZ
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=4, horizontal_spacing=0.5, vertical_spacing=0.5,
                                        pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.add(rt.Transmitter("bs", position=mi.Point3f(*map(float, config.BS_CENTER_M)), orientation=mi.Point3f(float(np.pi / 2), 0.0, 0.0)))
        scene.add(rt.Receiver("amr", position=mi.Point3f(6.0, 8.0, config.RX_HEIGHT_M)))
        p = rt.PathSolver()(scene, max_depth=1, los=True, specular_reflection=True, refraction=False)
        cam = rt.Camera(position=mi.Point3f(9, -9, 16), look_at=mi.Point3f(9, 5, 0))
        out = FIG / "00_sionna_scene_render.png"
        scene.render_to_file(camera=cam, filename=str(out), paths=p, resolution=[960, 600], num_samples=64)
        return out
    except Exception as e:
        print(f"  Sionna 장면 렌더 생략: {e}")
        return None
