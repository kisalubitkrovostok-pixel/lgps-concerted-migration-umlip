"""Concerted Li migration events: directed-graph model of Sato et al., Chem. Mater. 2026, 38, 287.

  python concerted_events.py runs/LGPS_mace_700K
  python concerted_events.py runs/LGPS_mace_700K --tau 2 3 5 --rcut 1.6 2.0 2.4     # robustness scan

Procedure (per Sato):
 1. The trajectory is cut into windows of length tau. A Li ion whose displacement within a window exceeds
    d_hop is a hop = a node (start position, end position, displacement vector).
 2. Directed edge j -> i if the END of hop j lies within r_cut of the START of hop i (minimum image,
    cell matrix taken from md.traj): ion j moves into the place that ion i has left.
 3. Maximal simple paths are enumerated by DFS; of every group of paths sharing > 50 % of nodes only the
    longest is kept (this also removes sub-paths). A path whose apparent displacement is < r_cut is a closed
    loop (zero net transport) and is counted separately, not as transport.
 4. Event size n = number of ions in a path (n = 1: isolated single-ion hop).

Several values of --d_hop / --tau / --rcut give a robustness scan (scan.csv); the FIRST combination is the
main one (summary.json, paths.csv, plot). Parameters: d_hop (default 1.4 A, as Sato for LGPS; check the displacement histogram), r_cut (default 2.0 A;
must be shorter than the first Li-Li neighbour distance - printed), tau (default: auto = shortest window in
which >= 50 % of Li hop, i.e. Sato's 0.7 t_random-walk criterion).

Also reported (Sato eqs 1, 3): D_corr = <|R_fin - R_init|^2>/(6 tau) over transport paths and
sigma_DG / sigma_NE = <n_paths> D_corr / (N_Li D*), an estimate of 1/H_R independent of the noisy COM MSD.

Outputs in <run>/concerted/: summary.json, paths.csv, size_hist.csv, scan.csv (if scanning), concerted.png
"""
import argparse, json, os, csv
import numpy as np
from ase.io.trajectory import Trajectory

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--mobile", default="Li")
p.add_argument("--d_hop", type=float, nargs="+", default=[1.4], help="A; several values = scan")
p.add_argument("--rcut", type=float, nargs="+", default=[2.0])
p.add_argument("--tau", type=float, nargs="+", default=None, help="ps; default auto (50%% of Li hop)")
p.add_argument("--skip_ps", type=float, default=0.0, help="discard the beginning of the trajectory")
p.add_argument("--overlap", type=int, default=1, help="window starts every tau/overlap (1 = non-overlapping)")
p.add_argument("--max_paths", type=int, default=200000, help="safety cap per window")
p.add_argument("--no_plot", action="store_true")
args = p.parse_args()

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e4e3de"

rd = args.run.rstrip("/")
out = os.environ["RESULT_DIR"]
os.makedirs(out, exist_ok=True)

# ---------------- load, unwrap, remove framework drift ----------------
with Trajectory(os.path.join(rd, "md.traj")) as tr:
    F = len(tr)
    a0 = tr[0]
    pos = np.empty((F, len(a0), 3))
    for k in range(F):
        pos[k] = tr[k].positions
cell = np.array(a0.cell)
inv = np.linalg.inv(cell)
sym = np.array(a0.get_chemical_symbols())
log = np.genfromtxt(os.path.join(rd, "log.csv"), delimiter=",", names=True)
dtf = float(log["time_ps"][1] - log["time_ps"][0])          # ps per saved frame


def mic(v):
    f = v @ inv
    f -= np.round(f)
    return f @ cell


d = mic(pos[1:] - pos[:-1])
pos = np.concatenate([pos[:1], pos[0] + np.cumsum(d, axis=0)])
fw = sym != args.mobile
pos -= (pos[:, fw].mean(axis=1) - pos[0, fw].mean(axis=0))[:, None, :]
skip = int(round(args.skip_ps / dtf))
li = pos[skip:, sym == args.mobile]
F, N = li.shape[:2]
T_run = (F - 1) * dtf
V = abs(np.linalg.det(cell))
widths = [V / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
print(f"{os.path.basename(rd)}: {F} frames x {dtf} ps = {T_run:.0f} ps, {N} {args.mobile}, "
      f"cell {np.round(np.linalg.norm(cell, axis=1), 3)} A, min width {min(widths):.2f} A")
for rc in args.rcut:
    if rc >= min(widths) / 2:
        raise SystemExit(f"r_cut {rc} >= half the smallest cell width ({min(widths)/2:.2f} A): minimum image breaks")

# ---------------- D* (long-time MSD) ----------------
lags = np.unique(np.linspace(1, (F - 1) // 2, 80).astype(int))
msd = np.array([np.mean(np.sum((li[L:] - li[:-L]) ** 2, axis=-1)) for L in lags])
tl = lags * dtf
win = tl >= min(5.0, T_run / 4)
D_self = np.polyfit(tl[win], msd[win], 1)[0] / 6          # A^2/ps
print(f"D*({args.mobile}) = {D_self * 1e-4:.3e} cm^2/s")

# ---------------- Li-Li nearest-neighbour distance (for r_cut sanity) ----------------
nn = []
for k in range(0, F, max(1, F // 60)):
    dd = np.linalg.norm(mic(li[k][:, None, :] - li[k][None, :, :]), axis=-1)
    np.fill_diagonal(dd, np.inf)
    nn.append(dd.min(axis=1))
nn = np.concatenate(nn)
h_nn, e_nn = np.histogram(nn, bins=np.arange(1.0, 4.0, 0.05))
nn_mode = float(e_nn[np.argmax(h_nn)] + 0.025)
print(f"Li-Li nearest neighbour: mode {nn_mode:.2f} A, 5th percentile {np.percentile(nn, 5):.2f} A")
for rc in args.rcut:
    if rc >= nn_mode:
        print(f"  WARNING: r_cut {rc} >= typical Li-Li neighbour distance -> static neighbours may be linked")

# ---------------- tau ----------------
D_HOP0 = args.d_hop[0]


def hop_fraction(L, stride=5):
    t0 = np.arange(0, F - L, stride)
    return float(np.mean(np.linalg.norm(li[t0 + L] - li[t0], axis=-1) > D_HOP0))


if args.tau:
    taus = [max(1, int(round(t / dtf))) for t in args.tau]
else:
    L = 1
    while L < (F - 1) // 4 and hop_fraction(L) < 0.5:
        L = int(L * 1.15) + 1
    taus = [L]
    print(f"tau (auto, >= 50 % of {args.mobile} hop): {L * dtf:.2f} ps "
          f"(hop fraction {hop_fraction(L):.2f})")


# ---------------- graph analysis ----------------
def maximal_paths(succ, h):
    paths = []
    for s in range(h):
        stack = [[s]]
        while stack:
            path = stack.pop()
            ext = [x for x in succ[path[-1]] if x not in path]
            if not ext:
                paths.append(path)
            else:
                stack.extend(path + [x] for x in ext)
            if len(paths) > args.max_paths:
                raise RuntimeError("too many paths in one window: lower r_cut or tau")
    return paths


cdir = cell[2] / np.linalg.norm(cell[2])     # channel axis (c) for direction analysis


def analyze(L, rc, dh, record=False):
    step = max(1, L // args.overlap)
    sizes, loop_sizes, rows = [], [], []
    n_hops = n_win = 0
    sum_dR2 = 0.0
    n_transport_paths = 0
    dR_all, n_all = [], []
    disp_all = []
    for t in range(0, F - L, step):
        r0, r1 = li[t], li[t + L]
        dv = r1 - r0
        dist = np.linalg.norm(dv, axis=-1)
        disp_all.append(dist)
        n_win += 1
        hop = np.where(dist > dh)[0]
        h = len(hop)
        n_hops += h
        if h == 0:
            continue
        S, E, Dv = r0[hop], r1[hop], dv[hop]
        gap = np.linalg.norm(mic(E[:, None, :] - S[None, :, :]), axis=-1)   # end_j vs start_i
        adj = gap < rc
        np.fill_diagonal(adj, False)
        succ = [np.flatnonzero(adj[j]).tolist() for j in range(h)]
        paths = sorted(maximal_paths(succ, h), key=len, reverse=True)
        kept, kept_sets = [], []
        for pth in paths:
            st = set(pth)
            if all(len(st & k) / len(st) <= 0.5 for k in kept_sets):
                kept.append(pth)
                kept_sets.append(st)
        for pth in kept:
            dR = Dv[pth[0]].copy()
            for a_, b_ in zip(pth[:-1], pth[1:]):
                dR += mic(S[b_] - E[a_]) + Dv[b_]
            mag = float(np.linalg.norm(dR))
            loop = len(pth) >= 2 and mag < rc
            (loop_sizes if loop else sizes).append(len(pth))
            if not loop:
                sum_dR2 += mag ** 2
                n_transport_paths += 1
                dR_all.append(dR)
                n_all.append(len(pth))
            if record:
                rows.append([round((t + skip) * dtf, 3), len(pth), int(loop), round(mag, 3),
                             *np.round(dR, 3), " ".join(str(int(hop[q])) for q in pth)])
    tau_ps = L * dtf
    sizes = np.array(sizes, int)
    nmax = max(sizes.max() if len(sizes) else 1, 1)
    Nn = np.bincount(sizes, minlength=nmax + 1)[1:]                       # number of events of size n
    ions_in = np.arange(1, nmax + 1) * Nn
    Pn = ions_in / ions_in.sum() if ions_in.sum() else ions_in            # Sato eq 4 (ion-weighted)
    D_corr = sum_dR2 / max(n_transport_paths, 1) / (6 * tau_ps)          # A^2/ps
    paths_per_win = n_transport_paths / n_win
    ratio = paths_per_win * D_corr / (N * D_self) if D_self > 0 else float("nan")
    dR_all = np.array(dR_all).reshape(-1, 3)
    n_all = np.array(n_all, int)
    mags = np.linalg.norm(dR_all, axis=1)
    cosc = np.abs(dR_all @ cdir) / np.maximum(mags, 1e-9)
    conc = n_all >= 2
    per_n = {int(k): {"count": int((n_all == k).sum()), "median_dR_A": round(float(np.median(mags[n_all == k])), 2),
                      "frac_along_c": round(float(np.mean(cosc[n_all == k] > 0.8)), 2)}
             for k in range(1, min(int(n_all.max()) if len(n_all) else 1, 12) + 1) if (n_all == k).any()}
    res = {
        "tau_ps": round(tau_ps, 3), "r_cut_A": rc, "d_hop_A": dh, "windows": n_win,
        "hops_total": int(n_hops), "hop_fraction_per_window": round(n_hops / (n_win * N), 3),
        "events_total": int(len(sizes)), "closed_loops": int(len(loop_sizes)),
        "events_per_ps": round(len(sizes) / (n_win * tau_ps), 3),
        "concerted_events_n_ge_2": int((sizes >= 2).sum()),
        "concerted_events_per_ps": round(float((sizes >= 2).sum()) / (n_win * tau_ps), 3),
        "mean_n_event_weighted": round(float(sizes.mean()) if len(sizes) else 0, 3),
        "mean_n_ion_weighted": round(float((np.arange(1, nmax + 1) * Pn).sum()), 3),
        "frac_hops_in_concerted": round(float(Pn[1:].sum()) if len(Pn) > 1 else 0.0, 3),
        "N_n": Nn.tolist(), "P_n_ion_weighted": np.round(Pn, 4).tolist(),
        "D_corr_cm2_s": float(f"{D_corr * 1e-4:.3e}"), "D_self_cm2_s": float(f"{D_self * 1e-4:.3e}"),
        "sigmaDG_over_sigmaNE": round(float(ratio), 3),
        "frac_along_c_singles": round(float(np.mean(cosc[~conc] > 0.8)) if (~conc).any() else 0, 3),
        "frac_along_c_concerted": round(float(np.mean(cosc[conc] > 0.8)) if conc.any() else 0, 3),
        "median_dR_per_ion_concerted_A": round(float(np.median(mags[conc] / n_all[conc])) if conc.any() else 0, 3),
        "per_n": per_n,
    }
    return res, rows, np.concatenate(disp_all)


scan = []
main = None
for dh in args.d_hop:
  for L in taus:
    for rc in args.rcut:
        rec = main is None
        res, rows, disp = analyze(L, rc, dh, record=rec)
        scan.append(res)
        print(f"d_hop {dh:.2f}  tau {res['tau_ps']:6.2f} ps  r_cut {rc:.2f}  hops {res['hops_total']:6d}  events {res['events_total']:6d}"
              f"  n>=2: {res['concerted_events_n_ge_2']:5d} ({res['concerted_events_per_ps']:.2f}/ps)"
              f"  <n>_ion {res['mean_n_ion_weighted']:.2f}  hops in concerted {res['frac_hops_in_concerted']:.2f}"
              f"  loops {res['closed_loops']}  sigDG/sigNE {res['sigmaDG_over_sigmaNE']:.2f}"
              f"  along c: singles {res['frac_along_c_singles']:.2f} / concerted {res['frac_along_c_concerted']:.2f}"
              f"  |dR|/n {res['median_dR_per_ion_concerted_A']:.2f} A")
        if rec:
            main, main_rows, main_disp = res, rows, disp

# ---------------- write ----------------
main.update({"run": os.path.basename(rd), "frames_used": int(F), "t_ps": round(T_run, 1), "n_Li": int(N),
             "Li_Li_nn_mode_A": nn_mode, "cell_A": np.round(cell, 5).tolist()})
json.dump(main, open(os.path.join(out, "summary.json"), "w"), indent=1)
with open(os.path.join(out, "paths.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_start_ps", "n_ions", "closed_loop", "dR_A", "dRx", "dRy", "dRz", "li_indices"])
    w.writerows(main_rows)
with open(os.path.join(out, "size_hist.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "N_events", "P_ion_weighted"])
    for n_, (a_, b_) in enumerate(zip(main["N_n"], main["P_n_ion_weighted"]), 1):
        w.writerow([n_, a_, b_])
if len(scan) > 1:
    with open(os.path.join(out, "scan.csv"), "w", newline="") as f:
        keys = ["d_hop_A", "tau_ps", "r_cut_A", "hops_total", "events_total", "concerted_events_n_ge_2", "concerted_events_per_ps",
                "mean_n_ion_weighted", "frac_hops_in_concerted", "closed_loops", "sigmaDG_over_sigmaNE",
                "frac_along_c_singles", "frac_along_c_concerted", "median_dR_per_ion_concerted_A"]
        w = csv.writer(f)
        w.writerow(keys)
        for r in scan:
            w.writerow([r[k] for k in keys])

# ---------------- plot ----------------
if not args.no_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax[0].hist(main_disp, bins=np.arange(0, max(6.0, np.percentile(main_disp, 99.5)), 0.05),
               color=SERIES[0], edgecolor="white", linewidth=0.3)
    for dh in args.d_hop:
        ax[0].axvline(dh, color=INK2, ls="--", lw=1)
        ax[0].text(dh, ax[0].get_ylim()[1] * 0.95, f" {dh}", color=INK2, va="top", fontsize=8)
    ax[0].set(xlabel=f"|Δr| of {args.mobile} within τ = {main['tau_ps']:.2f} ps (Å)", ylabel="count",
              title="Displacement distribution")
    n = np.arange(1, len(main["P_n_ion_weighted"]) + 1)
    ax[1].bar(n, main["P_n_ion_weighted"], color=SERIES[0], width=0.8)
    ax[1].set(xlabel="ions per event, n", ylabel="P(n), ion-weighted",
              title=f"Event size  (⟨n⟩ = {main['mean_n_ion_weighted']:.2f}; d_hop {main['d_hop_A']}, r_cut {main['r_cut_A']} Å)")
    ax[1].set_xticks(n[: min(len(n), 15)])
    for a in ax:
        a.grid(color=GRID, lw=0.6, axis="y")
        a.title.set_color(INK)
    fig.suptitle(os.path.basename(rd), color=INK, fontsize=10)
    fig.savefig(os.path.join(out, "concerted.png"), dpi=150)
print(f"-> {out}/")
