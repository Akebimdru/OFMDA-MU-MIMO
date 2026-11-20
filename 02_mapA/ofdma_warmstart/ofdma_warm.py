"""OFDMA 가변 대역: 직전 분배를 이어받는 warm start 알고리즘 (사용자 설계, 2026-11-20) vs 기존 방식 vs 정확한 최적.

사용: python ofdma_warm.py <맵 폴더 이름> <출력 폴더>
  예) python ofdma_warm.py sionna_mapB mapB
맵 폴더의 코드·데이터는 읽기만 하고, 결과는 이 폴더의 <출력 폴더>/ 에만 쓴다. 채널은 완벽한 CSI.

비교하는 방식 (모두 연속 대역 4개, 합 64, MRC, 0.1 s마다 결정)
* restart : 기존 방식. 매 지점 16개 균등에서 시작, 24가지 순서 각각 경계 이동, 최선 순서
* warm_greedy : 사용자 원안
    t=0  : 정확한 최적(이분 탐색)으로 k와 순서를 정한다.
    t>0  : 직전 구간 4개 그대로 두고 AMR별로 4개 구간 rate를 계산(4x4 표).
           "자기 최댓값이 가장 작은 AMR"부터 남은 구간 중 자기에게 가장 좋은 구간을 가져간다(매 라운드 다시 계산).
           그다음 경계 이동(최대 rate AMR → 최소 rate AMR로 한 칸씩 연쇄 이동, R_min이 엄격히 오를 때만).
* warm_exact : 위와 같고, 구간 배정만 4x4 표에서 24가지 배정 전수 (최저 rate 최대).
* exact : 매 지점 정확한 최적 (24 순서 × 목표 rate 이분 탐색) — 일치율 기준.

계산 횟수 (뺄셈 = 누적합 표에서 대역 rate 하나 구하기)
* restart : 순서마다 시작 4 + 시도마다 4 (받아들인 이동 + 마지막 거절 1번)
* warm    : 4x4 표 16 + 시도마다 4
대역 전환 (d)
* 순서 변경 : 어떤 AMR이든 대역 위치(왼쪽부터 몇 번째)가 바뀐 0.1 s 단계 수
* 주인 바뀐 부반송파 수 : 0.1 s마다 64개 중 담당 AMR이 바뀐 부반송파 개수의 합
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAP = sys.argv[1]
OUT = ROOT / sys.argv[2]
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent / MAP / "code"))

import numpy as np  # noqa: E402

import config  # noqa: E402
from collision import AMRS  # noqa: E402
from fine_eval import placements  # noqa: E402
from ofdma import PERMS  # noqa: E402  PERMS[p, k] = AMR k의 대역 위치
from ofdma_exact import allocation, exact  # noqa: E402
from ofdma_var import ORDERS, sc_prefix_table, setup  # noqa: E402  ORDERS[p, j] = 위치 j의 AMR

J = np.arange(1, 4)
TOL = 1e4   # 0.01 Mbps


def shift_loop(prefix, pts, b, r, cur):
    """경계 이동. pts (m,4) 위치별 점, b (m,5), r (m,4), cur (m,). 제자리 갱신, 시도 횟수 (m,) 반환."""
    m = len(cur)
    active = np.ones(m, bool)
    tries = np.zeros(m, np.int32)
    while active.any():
        ia = np.flatnonzero(active)
        tries[ia] += 1
        ra, ba = r[ia], b[ia]
        pmax, pmin = ra.argmax(1), ra.argmin(1)
        ok = np.diff(ba, axis=1)[np.arange(len(ia)), pmax] > 1
        lo, hi = np.minimum(pmax, pmin)[:, None], np.maximum(pmax, pmin)[:, None]
        sh = np.where((J > lo) & (J <= hi), np.where(pmax < pmin, -1, 1)[:, None], 0)
        nb = ba.copy()
        nb[:, 1:4] += sh
        p = pts[ia]
        nr = prefix[p, nb[:, 1:]] - prefix[p, nb[:, :-1]]
        new = nr.min(1)
        acc = ok & (new > cur[ia])
        g = ia[acc]
        b[g], r[g], cur[g] = nb[acc], nr[acc], new[acc]
        active[ia[~acc]] = False
    return tries


def restart_counted(prefix, cells4):
    """기존 방식 (ofdma_var.balance와 같은 계산) + 시도 횟수 합. 반환: R_min, 위치별 AMR (u,4), 경계 (u,5), 시도 합."""
    u = len(cells4)
    pts = cells4[:, ORDERS].reshape(-1, 4)
    b = np.tile(np.array([0, 16, 32, 48, 64]), (len(pts), 1))
    r = prefix[pts, b[:, 1:]] - prefix[pts, b[:, :-1]]
    cur = r.min(1)
    tries = shift_loop(prefix, pts, b, r, cur)
    cur = cur.reshape(u, 24)
    best = cur.argmax(1)
    sel = np.arange(u) * 24 + best
    return cur.max(1), ORDERS[best], b[sel], tries.reshape(u, 24).sum(1)


def owner_map(slot_amr, b):
    """(m,4) 위치별 AMR, (m,5) 경계 → (m,64) 부반송파별 담당 AMR."""
    n = np.arange(64)
    pos = (n[None, :, None] >= b[:, None, 1:4]).sum(2)              # (m,64) 위치
    return np.take_along_axis(slot_amr, pos, axis=1)


def assign_exact(tab):
    """tab (m, AMR 4, 위치 4) → 위치별 AMR (m,4)."""
    v = tab[:, np.arange(4)[None, :], PERMS].min(2)                 # (m,24)
    return ORDERS[v.argmax(1)]


def assign_greedy(tab):
    """자기 최댓값이 가장 작은 AMR부터 남은 위치 중 최선을 가져간다."""
    m = len(tab)
    rows = np.arange(m)
    slot_amr = np.full((m, 4), -1)
    amr_left = np.ones((m, 4), bool)
    slot_left = np.ones((m, 4), bool)
    for _ in range(4):
        t = np.where(slot_left[:, None, :], tab, -np.inf)
        best_slot = t.argmax(2)                                       # (m, AMR)
        best_val = np.where(amr_left, t.max(2), np.inf)
        k = best_val.argmin(1)
        s = best_slot[rows, k]
        slot_amr[rows, s] = k
        amr_left[rows, k] = False
        slot_left[rows, s] = False
    return slot_amr


def warm_run(prefix, P, start_amr, start_b, assign):
    """P (N,T,4) AMR별 점. t=0은 주어진 분배, 이후 warm start. 반환 dict of (N,T)."""
    N, T, _ = P.shape
    rows = np.arange(N)
    rmin = np.empty((N, T), np.float32)
    subs = np.zeros((N, T), np.int16)
    order_change = np.zeros((N, T), bool)
    sc_change = np.zeros((N, T), np.int8)
    slot_amr, b = start_amr.copy(), start_b.copy()
    pts = P[:, 0][rows[:, None], slot_amr]
    r = prefix[pts, b[:, 1:]] - prefix[pts, b[:, :-1]]
    rmin[:, 0] = r.min(1)
    own = owner_map(slot_amr, b)
    for ti in range(1, T):
        pk = P[:, ti]                                                 # (N,4) AMR별 점
        tab = prefix[pk[:, :, None], b[:, None, 1:]] - prefix[pk[:, :, None], b[:, None, :-1]]   # (N, AMR, 위치)
        new_amr = assign(tab)
        order_change[:, ti] = (new_amr != slot_amr).any(1)
        slot_amr = new_amr
        pts = pk[rows[:, None], slot_amr]
        r = tab[rows[:, None], slot_amr, np.arange(4)[None, :]]
        cur = r.min(1)
        tries = shift_loop(prefix, pts, b, r, cur)
        rmin[:, ti] = cur
        subs[:, ti] = 16 + 4 * tries
        o = owner_map(slot_amr, b)
        sc_change[:, ti] = (o != own).sum(1)
        own = o
    return {"rmin": rmin, "subs": subs, "order_change": order_change, "sc_change": sc_change}


def main():
    t_all = time.time()
    h_flat, paths, combos, t, seq, nx = setup()
    prefix = sc_prefix_table(h_flat)
    if len(sys.argv) > 3:                       # 시험 실행용: 앞에서 n개 조합만
        combos = combos[:int(sys.argv[3])]
    N, T = len(combos), len(t)
    P = np.stack([seq[amr][combos[:, k]] for k, amr in enumerate(AMRS)], axis=2).astype(np.int32)   # (N,T,4)
    c4, inv = placements(seq, combos)
    print(f"{MAP}: {N:,} combos × {T} steps = {N * T:,} decisions, {len(c4):,} unique placements", flush=True)

    # ---- 정확한 최적 (고유 배치마다) ----
    t0 = time.time()
    ex_val = np.empty(len(c4))
    ex_ord = np.empty(len(c4), np.int16)
    for a in range(0, len(c4), 200_000):
        v, o, _ = exact(prefix, c4[a:a + 200_000])
        ex_val[a:a + 200_000], ex_ord[a:a + 200_000] = v, o
        print(f"  exact {min(a + 200_000, len(c4)):,}/{len(c4):,}  {(time.time() - t0) / 60:.1f} min", flush=True)
    time_exact_dedup = time.time() - t0

    # ---- 기존 방식 (고유 배치마다) ----
    t0 = time.time()
    rs_val = np.empty(len(c4))
    rs_amr = np.empty((len(c4), 4), np.int8)
    rs_b = np.empty((len(c4), 5), np.int8)
    rs_tries = np.empty(len(c4), np.int32)
    for a in range(0, len(c4), 200_000):
        v, sa, bb, tr = restart_counted(prefix, c4[a:a + 200_000])
        rs_val[a:a + 200_000], rs_amr[a:a + 200_000], rs_b[a:a + 200_000], rs_tries[a:a + 200_000] = v, sa, bb, tr
    time_restart_dedup = time.time() - t0
    print(f"  restart done {time_restart_dedup / 60:.1f} min", flush=True)

    # ---- t=0 이상적 분배 (정확한 최적) ----
    s0, z0 = allocation(prefix, P[:, 0], ex_ord[inv[:, 0]].astype(int), ex_val[inv[:, 0]])
    start_amr = np.argsort(s0, axis=1)                                # 위치별 AMR
    start_b = np.concatenate([np.sort(s0, axis=1), np.full((N, 1), 64)], axis=1)

    res = {}
    for name, fn in [("warm_greedy", assign_greedy), ("warm_exact", assign_exact)]:
        t0 = time.time()
        res[name] = warm_run(prefix, P, start_amr, start_b, fn)
        res[name]["time"] = time.time() - t0
        print(f"  {name} {res[name]['time'] / 60:.1f} min", flush=True)

    # ---- 같은 표본에서 결정 1번당 시간 (중복 제거 없이) ----
    rng = np.random.default_rng(0)
    smp = rng.choice(N, min(500, N), replace=False)
    flat = P[smp].reshape(-1, 4)
    tt = time.time(); restart_counted(prefix, flat); us_restart = (time.time() - tt) / len(flat) * 1e6
    tt = time.time(); exact(prefix, flat); us_exact = (time.time() - tt) / len(flat) * 1e6
    us = {"restart": us_restart, "exact": us_exact}
    for name, fn in [("warm_greedy", assign_greedy), ("warm_exact", assign_exact)]:
        tt = time.time(); warm_run(prefix, P[smp], start_amr[smp], start_b[smp], fn)
        us[name] = (time.time() - tt) / len(flat) * 1e6

    # ---- 기존 방식의 시간축 지표 ----
    rs_inst = rs_val[inv]
    rs_subs = (24 * 4 + 4 * rs_tries)[inv]
    amr_t, b_t = rs_amr[inv], rs_b[inv]                              # (N,T,4), (N,T,5)
    rs_order_change = (amr_t[:, 1:] != amr_t[:, :-1]).any(2)
    rs_sc = np.zeros((N, T - 1), np.int16)
    for ti in range(1, T):
        rs_sc[:, ti - 1] = (owner_map(amr_t[:, ti], b_t[:, ti]) != owner_map(amr_t[:, ti - 1], b_t[:, ti - 1])).sum(1)
    ex_inst = ex_val[inv]

    mb = lambda v: round(float(v) / 1e6, 3)

    def pick(inst):
        rmin, rmean = inst.min(1), inst.mean(1)
        i = int(np.lexsort((rmean, rmin))[-1])
        return i, rmin

    summary = {"map": MAP, "combos": N, "steps": T, "decisions": N * T, "unique_placements": int(len(c4)), "methods": {}}
    traces = {}
    ex_i, ex_rmin = pick(ex_inst)
    for name, inst, subs, oc, scc in [
        ("restart (current)", rs_inst, rs_subs, rs_order_change, rs_sc),
        ("warm_greedy (user)", res["warm_greedy"]["rmin"], res["warm_greedy"]["subs"][:, 1:], res["warm_greedy"]["order_change"][:, 1:], res["warm_greedy"]["sc_change"][:, 1:]),
        ("warm_exact (24 assign)", res["warm_exact"]["rmin"], res["warm_exact"]["subs"][:, 1:], res["warm_exact"]["order_change"][:, 1:], res["warm_exact"]["sc_change"][:, 1:]),
    ]:
        i, rmin = pick(inst)
        gap = (ex_inst - inst) / 1e6
        key = name.split()[0]
        total_time = {"restart": time_restart_dedup, "warm_greedy": res["warm_greedy"]["time"], "warm_exact": res["warm_exact"]["time"]}[key]
        summary["methods"][name] = {
            "a_best_rmin_mbps": mb(rmin[i]), "a_best_paths": dict(zip(AMRS, map(int, combos[i]))),
            "a_median_rmin_over_combos_mbps": mb(np.median(rmin)),
            "a_rmin_gap_to_exact_on_same_combo_mbps": {"mean": round(float(((ex_rmin - rmin) / 1e6).mean()), 4),
                                                        "max": round(float(((ex_rmin - rmin) / 1e6).max()), 3),
                                                        "identical_pct": round(float((ex_rmin - rmin < TOL).mean() * 100), 2)},
            "b_match_exact_per_decision_pct": round(float((gap < 0.01).mean() * 100), 3),
            "b_gap_per_decision_mbps": {"mean": round(float(gap.mean()), 4), "p99": round(float(np.percentile(gap, 99)), 3), "max": round(float(gap.max()), 3)},
            "c_subtractions_per_decision": {"mean": round(float(subs.mean()), 1), "p90": float(np.percentile(subs, 90)), "max": int(subs.max())},
            "d_order_changes_per_trajectory": {"mean": round(float(oc.sum(1).mean()), 1), "best_combo": int(oc[i].sum())},
            "d_subcarrier_owner_changes_per_trajectory": {"mean": round(float(scc.sum(1).mean()), 1), "best_combo": int(scc[i].sum()),
                                                          "per_step_mean": round(float(scc.mean()), 2)},
            "time_total_s": round(total_time, 1),
            "time_note": "restart: per unique placement (dedup); warm: every decision, no dedup",
            "us_per_decision_same_sample": round(us[key], 2),
            "time_no_dedup_est_min": round(us[key] * N * T / 6e7, 1),
        }
        traces[key] = (i, inst[i])
    summary["exact"] = {"best_rmin_mbps": mb(ex_rmin[ex_i]), "best_paths": dict(zip(AMRS, map(int, combos[ex_i]))),
                        "time_total_dedup_s": round(time_exact_dedup, 1), "us_per_decision_same_sample": round(us["exact"], 2),
                        "time_no_dedup_est_min": round(us["exact"] * N * T / 6e7, 1)}
    summary["warm_assign_comparisons_per_decision"] = {"greedy": "4 rounds over remaining 4x4 entries (16+12+8+4 max/argmax)",
                                                       "exact_24": "24 x 4 lookups + 24 mins (no extra rate computation)"}
    summary["runtime_total_min"] = round((time.time() - t_all) / 60, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(OUT / "results.npy", {"combos": combos, "exact_rmin": ex_rmin, "restart_rmin": rs_inst.min(1),
                                  "warm_greedy_rmin": res["warm_greedy"]["rmin"].min(1), "warm_exact_rmin": res["warm_exact"]["rmin"].min(1)},
            allow_pickle=True)
    plot(summary, t, ex_inst, rs_inst, res, rs_subs, rs_order_change)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def plot(summary, t, ex_inst, rs_inst, res, rs_subs, rs_oc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["restart", "warm_greedy", "warm_exact"]
    labels = ["restart (current)", "warm start + greedy assign (user)", "warm start + 24 assign"]
    colors = ["tab:gray", "tab:blue", "tab:orange"]
    insts = [rs_inst, res["warm_greedy"]["rmin"], res["warm_exact"]["rmin"]]
    subs = [rs_subs, res["warm_greedy"]["subs"][:, 1:], res["warm_exact"]["subs"][:, 1:]]
    ms = list(summary["methods"].values())
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    ex_r = ex_inst.min(1) / 1e6
    for n, lab, c, inst in zip(names, labels, colors, insts):
        a.hist(inst.min(1) / 1e6, bins=60, histtype="step", color=c, lw=1.5, label=f"{lab}: best {inst.min(1).max() / 1e6:.2f}")
    a.hist(ex_r, bins=60, histtype="step", color="k", lw=1, ls="--", label=f"exact: best {ex_r.max():.2f}")
    a.set_xlabel("R_min per path combination [Mbps]"); a.set_ylabel("# combos"); a.legend(fontsize=8); a.grid(alpha=.3)
    a.set_title("(a) R_min over all collision-free path combinations")
    a = ax[0, 1]
    for lab, c, inst in zip(labels, colors, insts):
        g = np.sort(((ex_inst - inst) / 1e6).ravel())
        a.plot(g[::max(1, len(g) // 20000)], np.linspace(0, 100, len(g[::max(1, len(g) // 20000)])), color=c, label=lab)
    a.set_xscale("symlog", linthresh=0.01); a.set_xlabel("exact − method, per 0.1 s decision [Mbps]"); a.set_ylabel("CDF [%]")
    a.legend(fontsize=8); a.grid(alpha=.3)
    a.set_title("(b) " + ", ".join(f"{n}: {m['b_match_exact_per_decision_pct']:.1f}% exact" for n, m in zip(["restart", "greedy", "warm24"], ms)), fontsize=9)
    a = ax[1, 0]
    x = np.arange(3)
    a.bar(x - 0.2, [m["c_subtractions_per_decision"]["mean"] for m in ms], 0.4, color=colors, label="subtractions / decision (mean)")
    for i, m in enumerate(ms):
        a.text(i - 0.2, m["c_subtractions_per_decision"]["mean"], f"{m['c_subtractions_per_decision']['mean']:.0f}", ha="center", va="bottom")
    a2 = a.twinx()
    a2.bar(x + 0.2, [m["us_per_decision_same_sample"] for m in ms], 0.4, color=colors, alpha=0.45, hatch="//")
    for i, m in enumerate(ms):
        a2.text(i + 0.2, m["us_per_decision_same_sample"], f"{m['us_per_decision_same_sample']:.1f} µs", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, ["restart", "warm+greedy", "warm+24"]); a.set_ylabel("subtractions / decision (solid)"); a2.set_ylabel("µs / decision (hatched)")
    a.set_title("(c) cost per 0.1 s decision"); a.grid(alpha=.3, axis="y")
    a = ax[1, 1]
    a.bar(x - 0.2, [m["d_order_changes_per_trajectory"]["mean"] for m in ms], 0.4, color=colors)
    a.bar(x + 0.2, [m["d_subcarrier_owner_changes_per_trajectory"]["mean"] / 10 for m in ms], 0.4, color=colors, alpha=.45, hatch="//")
    for i, m in enumerate(ms):
        a.text(i - 0.2, m["d_order_changes_per_trajectory"]["mean"], f"{m['d_order_changes_per_trajectory']['mean']:.0f}", ha="center", va="bottom", fontsize=8)
        a.text(i + 0.2, m["d_subcarrier_owner_changes_per_trajectory"]["mean"] / 10, f"{m['d_subcarrier_owner_changes_per_trajectory']['mean']:.0f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, ["restart", "warm+greedy", "warm+24"])
    a.set_ylabel("per trajectory (mean over combos)")
    a.set_title("(d) band-order changes (solid) / subcarrier owner changes ÷10 (hatched)"); a.grid(alpha=.3, axis="y")
    fig.suptitle(f"OFDMA variable band: warm start vs restart vs exact — {summary['map']} ({summary['combos']:,} combos × {summary['steps']} steps)")
    fig.tight_layout()
    fig.savefig(OUT / "warm_vs_restart.png", dpi=130)


if __name__ == "__main__":
    main()
