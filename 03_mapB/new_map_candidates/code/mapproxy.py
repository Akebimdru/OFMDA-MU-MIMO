"""Fast proxy for comparing candidate maps before Sionna RT.

Channel: LoS + 6 first-order image reflections (4 walls, floor, ceiling), spherical wave per
BS antenna and subcarrier. Box obstacles (with height) block segments:
  blocked LoS -> -20 dB (edge diffraction stand-in), blocked reflection -> -40 dB.
Rates: OFDMA variable (continuous split, R=64/sum(1/r_i)), MU-MIMO 4x4 ZF, hybrid 2x2 (best
pairing, continuous split, 4x2 ZF inside each band). Robots follow simultaneous paths at 0.5 m/s.
"""
from __future__ import annotations
import heapq, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

C = 299_792_458.0
FC, BW, K = 5.8e9, 20e6, 64
DF = BW / K
FK = FC + (np.arange(K) - K / 2) * DF
LAM = C / FC
P_DBM, NF = -5.0, 7.0
N_DBM = -174 + 10 * np.log10(DF) + NF
SNR_SCALE = 10 ** ((P_DBM - N_DBM) / 10)
RX_H = 0.5
GAMMA = {"wall": 0.5, "floor": 0.5, "ceiling": 0.5}
LOSS_BLOCK = {"metal": 10 ** (-20 / 20), "concrete": 10 ** (-20 / 20)}
LOSS_BLOCK_REFL = 10 ** (-40 / 20)


def bs_ant(bs):
    p = np.repeat(np.asarray(bs, float)[None], 4, 0)
    p[:, 0] += (np.arange(4) - 1.5) * LAM / 2
    return p


def seg_hits_box(a, b, box):
    """a,b (...,3). box (x0,x1,y0,y1,h). slab test, returns bool (...)."""
    x0, x1, y0, y1, h = box
    lo = np.array([x0, y0, 0.0]); hi = np.array([x1, y1, h])
    d = b - a
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - a) / d; t2 = (hi - a) / d
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # parallel and outside slab
    par = d == 0
    outside = par & ((a < lo) | (a > hi))
    tmin = np.where(par, -np.inf, tmin); tmax = np.where(par, np.inf, tmax)
    te = np.max(tmin, -1); tx = np.min(tmax, -1)
    return (te <= tx) & (tx > 1e-6) & (te < 1 - 1e-6) & ~np.any(outside, -1)


def seg_hits_cyl(a, b, cyl):
    """full-height vertical cylinder (cx, cy, r, h): 2D distance of XY segment to axis < r (and z below h)."""
    cx, cy, r, h = cyl
    p = np.array([cx, cy])
    a2, b2 = a[..., :2], b[..., :2]
    d = b2 - a2
    L2 = np.sum(d * d, -1)
    t = np.clip(np.sum((p - a2) * d, -1) / np.where(L2 == 0, 1, L2), 0, 1)
    q = a2 + t[..., None] * d
    zq = a[..., 2] + t * (b[..., 2] - a[..., 2])
    return (np.linalg.norm(q - p, axis=-1) < r) & (zq < h) & (t > 1e-6) & (t < 1 - 1e-6)


class Layout:
    def __init__(self, spec):
        self.s = spec
        self.X, self.Y, self.Z = spec["room"]
        self.boxes = spec["boxes"]  # list of dict
        self.ant = bs_ant(spec["bs"])

    def blocked_factor(self, a, b, refl=False):
        f = np.ones(np.broadcast_shapes(a.shape, b.shape)[:-1])
        for bx in self.boxes:
            if "cx" in bx:
                hit = seg_hits_cyl(a, b, (bx["cx"], bx["cy"], bx["r"], bx["h"]))
            else:
                hit = seg_hits_box(a, b, (bx["x0"], bx["x1"], bx["y0"], bx["y1"], bx["h"]))
            f = np.where(hit, f * (LOSS_BLOCK_REFL if refl else LOSS_BLOCK[bx["mat"]]), f)
        return f

    def channel(self, xy):
        """xy (N,2) -> H (N,4,K) complex (normalised so |h|^2*SNR_SCALE = SNR)."""
        rx = np.c_[xy, np.full(len(xy), RX_H)]
        H = np.zeros((len(xy), 4, K), complex)
        planes = [(0, 0.0, "wall"), (0, self.X, "wall"), (1, 0.0, "wall"), (1, self.Y, "wall"),
                  (2, 0.0, "floor"), (2, self.Z, "ceiling")]
        for m, tx in enumerate(self.ant):
            tx3 = np.broadcast_to(tx, rx.shape)
            # LoS
            d = np.linalg.norm(rx - tx3, axis=1)
            g = LAM / (4 * np.pi * d) * self.blocked_factor(tx3, rx)
            H[:, m] += g[:, None] * np.exp(-2j * np.pi * FK[None] * d[:, None] / C)
            for ax, val, kind in planes:
                img = tx.copy(); img[ax] = 2 * val - img[ax]
                img3 = np.broadcast_to(img, rx.shape)
                dvec = rx - img3
                t = (val - img[ax]) / dvec[:, ax]
                p = img3 + t[:, None] * dvec
                dd = np.linalg.norm(dvec, axis=1)
                f = self.blocked_factor(tx3, p, True) * self.blocked_factor(p, rx, True)
                g = GAMMA[kind] * LAM / (4 * np.pi * dd) * f
                H[:, m] += g[:, None] * np.exp(-2j * np.pi * FK[None] * dd[:, None] / C)
        # diffuse scattering stand-in (Rician K = 15 dB w.r.t. specular power), deterministic per position
        pw = np.mean(np.abs(H) ** 2, axis=(1, 2))
        seed = (np.round(xy[:, 0] * 100).astype(np.int64) * 100003 + np.round(xy[:, 1] * 100).astype(np.int64))
        D = np.zeros_like(H)
        for i, sd in enumerate(seed):
            rng = np.random.default_rng(int(sd) % (2**32))
            n_taps = 8  # random delays up to 100 ns -> frequency selective
            tau = rng.uniform(0, 100e-9, n_taps)
            g = (rng.normal(size=(4, n_taps)) + 1j * rng.normal(size=(4, n_taps))) / np.sqrt(2 * n_taps)
            D[i] = g @ np.exp(-2j * np.pi * FK[None] * tau[:, None])
        H = H + D * np.sqrt(pw * 10 ** (-5 / 10))[:, None, None]
        return H * np.sqrt(SNR_SCALE)

    def footprint_blocked(self, xs, ys, r=0.25):
        X, Y = np.meshgrid(xs, ys)
        b = (X < r) | (X > self.X - r) | (Y < r) | (Y > self.Y - r)
        for bx in self.boxes:
            if "cx" in bx:
                b |= np.hypot(X - bx["cx"], Y - bx["cy"]) - bx["r"] <= r
                continue
            dx = np.maximum(np.maximum(bx["x0"] - X, X - bx["x1"]), 0)
            dy = np.maximum(np.maximum(bx["y0"] - Y, Y - bx["y1"]), 0)
            b |= np.hypot(dx, dy) <= r
        return b


# ---------------------------------------------------------------- rates
def mrc_snr(H):  # (...,4,K) -> (...,K)
    return np.sum(np.abs(H) ** 2, axis=-2)


CSI = {"nmse_db": None}
_RNG = np.random.default_rng(7)


def zf_snr(Hs, Hhat=None):  # Hs (U,4,K) -> (U,K) SINR with ZF built from Hhat (default: perfect CSI)
    Hk = np.transpose(Hs, (2, 1, 0))  # K,4,U
    if Hhat is None:
        G = np.conj(np.transpose(Hk, (0, 2, 1))) @ Hk
        G = G + 1e-9 * np.trace(G, axis1=1, axis2=2).real[:, None, None] * np.eye(G.shape[-1])
        inv = np.linalg.inv(G)
        return 1.0 / np.real(np.diagonal(inv, axis1=1, axis2=2)).T
    Hh = np.transpose(Hhat, (2, 1, 0))
    W = np.linalg.pinv(Hh)            # K,U,4
    A = np.abs(W @ Hk) ** 2           # K,U,U  (rows: stream k, cols: user j)
    sig = np.diagonal(A, axis1=1, axis2=2)
    interf = A.sum(-1) - sig
    noise = np.sum(np.abs(W) ** 2, -1)
    return (sig / (interf + noise)).T


def with_error(Hs):
    if CSI["nmse_db"] is None:
        return None
    p = np.mean(np.abs(Hs) ** 2, axis=(1, 2), keepdims=True)
    e = (_RNG.normal(size=Hs.shape) + 1j * _RNG.normal(size=Hs.shape)) / np.sqrt(2)
    return Hs + e * np.sqrt(p * 10 ** (CSI["nmse_db"] / 10))


def rate_per_sc(snr):  # Mbps per subcarrier averaged
    return DF * np.mean(np.log2(1 + snr), axis=-1) / 1e6


def scheme_rates(Hs):
    """Hs (4,4,K). returns min rate [Mbps] for ofdma_var, mumimo, hybrid."""
    r = rate_per_sc(mrc_snr(Hs))  # per-SC rate if alone
    ofdma = K / np.sum(1 / r)
    Hh = with_error(Hs)
    zf = zf_snr(Hs, Hh)
    mumimo = np.min(K * rate_per_sc(zf))
    best = 0
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        p = np.min(rate_per_sc(zf_snr(Hs[[a, b]], None if Hh is None else Hh[[a, b]])))
        q = np.min(rate_per_sc(zf_snr(Hs[[c, d]], None if Hh is None else Hh[[c, d]])))
        best = max(best, K * p * q / (p + q))
    return ofdma, mumimo, best


# ---------------------------------------------------------------- paths
MOVES = [(-1, -1, 2 ** .5), (-1, 0, 1), (-1, 1, 2 ** .5), (0, -1, 1), (0, 1, 1), (1, -1, 2 ** .5), (1, 0, 1), (1, 1, 2 ** .5)]


def astar(blocked, s, g, penalty=None):
    pen = np.zeros(blocked.shape) if penalty is None else penalty
    fr = [(0, 0, s)]; came = {s: None}; cost = {s: 0}; n = 0
    while fr:
        _, _, cur = heapq.heappop(fr)
        if cur == g:
            break
        for dr, dc, w in MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < blocked.shape[0] and 0 <= nb[1] < blocked.shape[1]) or blocked[nb]:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue
            c2 = cost[cur] + w * (1 + pen[nb])
            if c2 < cost.get(nb, 1e18):
                cost[nb] = c2; came[nb] = cur; n += 1
                heapq.heappush(fr, (c2 + np.hypot(nb[0] - g[0], nb[1] - g[1]), n, nb))
    path = [g]
    while came[path[-1]] is not None:
        path.append(came[path[-1]])
    return path[::-1]


def resample(poly, speed=0.5, dt=0.1):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1] / speed + dt / 2, dt)
    q = np.minimum(t * speed, s[-1])
    return np.c_[np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], s[-1]


def run(spec, out_png, res=0.25, snr_thr_db=None):
    L = Layout(spec)
    xs = np.arange(res / 2, L.X, res); ys = np.arange(res / 2, L.Y, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel()]
    H = L.channel(pts).reshape(len(ys), len(xs), 4, K)
    snr_db = 10 * np.log10(np.mean(mrc_snr(H), -1))
    # ULA direction cosine u = dx / r (what a 4-el x-ULA can resolve)
    bs = np.asarray(spec["bs"])
    r3 = np.sqrt((XX - bs[0]) ** 2 + (YY - bs[1]) ** 2 + (bs[2] - RX_H) ** 2)
    u = (XX - bs[0]) / r3
    # path grid 0.5 m
    gres = 0.5
    gx = np.arange(gres / 2, L.X, gres); gy = np.arange(gres / 2, L.Y, gres)
    gb = L.footprint_blocked(gx, gy)
    def cell(p):
        free = np.argwhere(~gb)
        d = (gx[free[:, 1]] - p[0]) ** 2 + (gy[free[:, 0]] - p[1]) ** 2
        return tuple(free[np.argmin(d)])
    # SNR at path-grid cells (nearest fine cell)
    snr_g = snr_db[np.clip((gy / res).astype(int), 0, len(ys) - 1)][:, np.clip((gx / res).astype(int), 0, len(xs) - 1)]
    thr = snr_thr_db if snr_thr_db is not None else spec.get("shadow_db", np.percentile(snr_db, 15))
    pen = np.where(snr_g < thr, 3.0, 0.0)
    res_out = {"name": spec["name"]}
    paths = {}
    for mode, P in (("shortest", None), ("snr_aware", pen)):
        trajs, lens = [], []
        for i, (s, g) in enumerate(zip(spec["starts"], spec["goals"])):
            cp = astar(gb, cell(s), cell(g), P)
            poly = [s] + [(gx[c[1]], gy[c[0]]) for c in cp] + [g]
            tr, ln = resample(poly)
            trajs.append(tr); lens.append(ln)
        paths[mode] = trajs
        T = max(len(t) for t in trajs)
        pos = np.stack([np.r_[t, np.repeat(t[-1:], T - len(t), 0)] for t in trajs])  # 4,T,2
        Ht = L.channel(pos.reshape(-1, 2)).reshape(4, T, 4, K)
        rates = np.array([scheme_rates(Ht[:, k]) for k in range(T)])  # T,3
        dmin = np.min([np.linalg.norm(pos[i] - pos[j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
        rates[dmin < 0.5] = np.nan   # collision instants (proxy has no collision avoidance)
        shadow_frac = [float(np.mean(10 * np.log10(np.mean(mrc_snr(Ht[i]), -1)) < thr)) for i in range(4)]
        res_out[mode] = {
            "travel_s": [round(l / 0.5, 1) for l in lens],
            "rmin": dict(zip(["ofdma_var", "mumimo", "hybrid"], np.round(np.nanmin(rates, 0), 1).tolist())),
            "median_inst_min": dict(zip(["ofdma_var", "mumimo", "hybrid"], np.round(np.nanmedian(rates, 0), 1).tolist())),
            "frac_time_hybrid_beats_mumimo": round(float(np.nanmean(np.where(np.isnan(rates[:,1]), np.nan, rates[:, 2] > rates[:, 1]))), 3),
            "frac_time_ofdma_beats_mumimo": round(float(np.nanmean(np.where(np.isnan(rates[:,1]), np.nan, rates[:, 0] > rates[:, 1]))), 3),
            "shadow_time_frac": [round(x, 2) for x in shadow_frac],
        }
        res_out[mode + "_rates"] = rates
    se = [snr_db[int(p[1] / res), int(p[0] / res)] for p in list(spec["starts"])]
    res_out["start_goal_snr_db"] = [round(float(x), 1) for x in se]
    res_out["shadow_thr_db"] = round(float(thr), 1)
    res_out["snr_median_db"] = round(float(np.median(snr_db)), 1)
    res_out["snr_p5_db"] = round(float(np.percentile(snr_db[~L.footprint_blocked(xs, ys, 0.0)], 5)), 1)
    res_out["paths"] = paths
    plot(L, spec, xs, ys, snr_db, u, paths, res_out, thr, out_png)
    return res_out


def plot(L, spec, xs, ys, snr_db, u, paths, R, thr, out_png):
    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.62], width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, :])
    occ = L.footprint_blocked(xs, ys, 0.0)
    im = ax.imshow(np.ma.masked_where(occ, snr_db), origin="lower", extent=(0, L.X, 0, L.Y), cmap="viridis",
                   vmin=np.percentile(snr_db, 2), vmax=np.percentile(snr_db, 99.5))
    plt.colorbar(im, ax=ax, fraction=0.025, label="MRC SNR [dB] (proxy)")
    ax.contour(xs, ys, snr_db, levels=[thr], colors="w", linewidths=0.8, linestyles=":")
    ax2.imshow(np.ma.masked_where(occ, u), origin="lower", extent=(0, L.X, 0, L.Y), cmap="twilight", vmin=-1, vmax=1)
    ax2.contour(xs, ys, u, levels=np.arange(-0.75, 0.8, 0.25), colors="k", linewidths=0.5)
    for a in (ax, ax2):
        for bx in L.boxes:
            fc = "silver" if bx["mat"] == "metal" else "dimgray"
            if "cx" in bx:
                a.add_patch(Circle((bx["cx"], bx["cy"]), bx["r"], fc="lightsteelblue", ec="k", lw=0.6)); continue
            a.add_patch(Rectangle((bx["x0"], bx["y0"]), bx["x1"] - bx["x0"], bx["y1"] - bx["y0"], fc=fc, ec="k", lw=0.6))
            if a is ax:
                a.text((bx["x0"] + bx["x1"]) / 2, (bx["y0"] + bx["y1"]) / 2, f"{bx['h']:g}m", ha="center", va="center", fontsize=7)
        a.plot(*spec["bs"][:2], "r^", ms=11, mec="w")
        for i, (s, g) in enumerate(zip(spec["starts"], spec["goals"])):
            a.add_patch(Circle(s, 0.25, fill=False, ec=f"C{i}", lw=2)); a.plot(*g, "*", color=f"C{i}", ms=13, mec="k", mew=.5)
            a.text(s[0], s[1] + 0.35, f"AMR{i+1}", color=f"C{i}", ha="center", fontsize=9, fontweight="bold",
                   bbox=dict(fc="w", alpha=.6, lw=0, pad=.5))
        a.set_xlim(0, L.X); a.set_ylim(0, L.Y); a.set_aspect("equal")
    for i in range(4):
        ax.plot(*paths["shortest"][i].T, "-", color=f"C{i}", lw=1.8)
        ax.plot(*paths["snr_aware"][i].T, "--", color=f"C{i}", lw=1.2)
    ax.set_title(f"{spec['name']}: SNR proxy, shortest (solid) vs shadow-avoiding (dashed); dotted = {thr:.0f} dB")
    ax2.set_title("ULA direction cosine u = dx/r (same contour band = hard for ZF)")
    t = np.arange(len(R["shortest_rates"])) * 0.1
    for j, (nm, c) in enumerate(zip(["var OFDMA", "4x4 MU-MIMO", "2x2 hybrid"], ["C0", "C3", "C2"])):
        ax3.plot(t, R["shortest_rates"][:, j], color=c, label=f"{nm} (shortest)")
        t2 = np.arange(len(R["snr_aware_rates"])) * 0.1
        ax3.plot(t2, R["snr_aware_rates"][:, j], color=c, ls="--", lw=0.9, label=f"{nm} (shadow-avoiding)")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("instantaneous min rate [Mbps]"); ax3.grid(alpha=.3)
    ax3.legend(ncol=3, fontsize=8)
    ax3.set_title("Proxy instantaneous R_min along simultaneous paths (no collision avoidance)")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
