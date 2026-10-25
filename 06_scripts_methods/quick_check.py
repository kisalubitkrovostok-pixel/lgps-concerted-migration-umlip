""Этот скрипт не особо важен для финального проекта, но он умеет проверять не поплыла ли ячейка. В m3gnet и еще одном потенциале поплыла. На mace и sevennnet - нет.

  python quick_check.py runs/LGPS_m3gnet_700K runs/LGPS_m3gnet_900K runs/LGPS_m3gnet_1100K

Works on finished AND still-running trajectories (reads whatever frames exist).
For every run directory writes  check.png  and  check.json, and prints a summary table:

  T_mean / T_std       thermostat sanity (expected std ~ T*sqrt(2/(3N)))
  Epot drift           eV/atom/ns over production - should be ~0 after equilibration
  MSD_framework(plat)  plateau of the framework MSD (A^2); if it keeps growing the framework is melting
  D_Li*                tracer D of Li (cm^2/s), linear fit of MSD in [fit_start, t_tot/2]
  D_sigma              Li centre-of-mass diffusivity (very noisy from one trajectory - indicative only)
  moved>2A             fraction of Li that ever moved > 2 A from its starting point
Positions are unwrapped by minimum image between consecutive frames (safe for 50 fs frames),
and the framework centre-of-mass drift is removed.
"""
import argparse, json, os, sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--mobile", default="Li")
p.add_argument("--fit_start_ps", type=float, default=5.0)
p.add_argument("--no_plot", action="store_true")
args = p.parse_args()

from ase.io.trajectory import Trajectory

# reference categorical palette (fixed order, never cycled); text stays neutral ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e4e3de"


def load(rd):
    with Trajectory(os.path.join(rd, "md.traj")) as tr:
        n = len(tr)
        a0 = tr[0]
        pos = np.empty((n, len(a0), 3))
        for i in range(n):
            pos[i] = tr[i].positions
    log = np.genfromtxt(os.path.join(rd, "log.csv"), delimiter=",", names=True)
    return a0, pos, log


def unwrap(pos, cell):
    inv = np.linalg.inv(cell)
    d = pos[1:] - pos[:-1]
    f = d @ inv
    f -= np.round(f)
    d = f @ cell
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1:] = pos[0] + np.cumsum(d, axis=0)
    return out


def msd_curve(x, lags):
    """x: (F, N, 3). Average over all time origins and particles."""
    return np.array([np.mean(np.sum((x[L:] - x[:-L]) ** 2, axis=-1)) for L in lags])


def com_msd(x, lags):
    c = x.mean(axis=1)                       # (F, 3)
    return np.array([np.mean(np.sum((c[L:] - c[:-L]) ** 2, axis=-1)) for L in lags])


rows = []
for rd in args.runs:
    rd = rd.rstrip("/")
    info = json.load(open(os.path.join(rd, "info.json"))) if os.path.exists(os.path.join(rd, "info.json")) else {}
    a0, pos, log = load(rd)
    F, N = pos.shape[:2]
    if F < 50:
        print(f"{rd}: only {F} frames, skip")
        continue
    dt_frame_ps = float(log["time_ps"][1] - log["time_ps"][0]) if len(log) > 1 else 0.05
    cell = np.array(a0.cell)
    sym = np.array(a0.get_chemical_symbols())
    pos = unwrap(pos, cell)
    fw = sym != args.mobile
    pos = pos - (pos[:, fw].mean(axis=1) - pos[0, fw].mean(axis=0))[:, None, :]   # remove framework drift

    t_tot = (F - 1) * dt_frame_ps
    lags = np.unique(np.linspace(1, (F - 1) // 2, 120).astype(int))
    tl = lags * dt_frame_ps
    species = [args.mobile] + sorted(set(sym) - {args.mobile})
    msd = {s: msd_curve(pos[:, sym == s], lags) for s in species}
    li = pos[:, sym == args.mobile]
    nli = li.shape[1]

    win = tl >= min(args.fit_start_ps, t_tot / 4)
    slope = np.polyfit(tl[win], msd[args.mobile][win], 1)[0]          # A^2/ps
    D_tr = slope / 6 * 1e-4                                            # cm^2/s
    cm = com_msd(li, lags)
    D_sig = np.polyfit(tl[win], cm[win], 1)[0] / 6 * 1e-4 * nli        # N * D_com
    fw_plateau = {s: float(np.mean(msd[s][win])) for s in species[1:]}
    fw_slope = {s: float(np.polyfit(tl[win], msd[s][win], 1)[0]) for s in species[1:]}   # A^2/ps
    moved = float(np.mean(np.max(np.linalg.norm(li - li[0], axis=-1), axis=0) > 2.0))
    # framework stability: look at the LEAST mobile framework species (e.g. Ge/P in LGPS, Zr/La in LLZO).
    # Anions (S, O) may rotate with their polyhedra; if even the stiffest species flows, the framework is melting.
    stiff = min(fw_slope, key=fw_slope.get)
    stiff_slope = fw_slope[stiff]
    xs = pos[:, sym == stiff]
    stiff_moved = float(np.mean(np.max(np.linalg.norm(xs - xs[0], axis=-1), axis=0) > 2.0))
    if stiff_slope < 0.002 and stiff_moved < 0.05:
        fw_status = "OK"
    elif stiff_slope < 0.05:
        fw_status = "FLOW"      # framework partially disordering - Li D not reliable
    else:
        fw_status = "MELTED"    # framework liquid - Li D is liquid diffusion, discard

    T = log["T_inst_K"]
    E = log["Epot_eV"]
    tE = log["time_ps"]
    drift = np.polyfit(tE, E, 1)[0] * 1000 / N if len(E) > 10 else float("nan")   # eV/atom/ns
    Tset = info.get("T", float(np.mean(T)))
    res = {
        "run": os.path.basename(rd), "frames": int(F), "t_ps": round(t_tot, 1), "natoms": int(N), "n_Li": int(nli),
        "T_mean": round(float(np.mean(T)), 1), "T_std": round(float(np.std(T)), 1),
        "T_std_expected": round(float(Tset * np.sqrt(2 / (3 * N))), 1),
        "Epot_drift_eV_atom_ns": round(float(drift), 4),
        "framework_msd_plateau_A2": {k: round(v, 3) for k, v in fw_plateau.items()},
        "framework_msd_slope_A2_ps": {k: round(v, 5) for k, v in fw_slope.items()},
        "D_tracer_cm2_s": float(f"{D_tr:.3e}"), "D_sigma_cm2_s": float(f"{D_sig:.3e}"),
        "Haven_ratio_rough": round(float(D_tr / D_sig), 2) if D_sig > 0 else None,
        "MSD_Li_final_A2": round(float(msd[args.mobile][-1]), 1),
        "frac_Li_moved_gt_2A": round(moved, 3),
        "framework_status": fw_status, "stiffest_species": stiff,
        "stiffest_frac_moved_gt_2A": round(stiff_moved, 3),
        "fit_window_ps": [round(float(tl[win][0]), 2), round(float(tl[-1]), 2)],
    }
    json.dump(res, open(os.path.join(rd, "check.json"), "w"), indent=1)
    rows.append(res)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                             "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
        for i, s in enumerate(species):
            ax[0].plot(tl, msd[s], color=SERIES[i % len(SERIES)], lw=1.8, label=s)
        ax[0].plot(tl[win], slope * tl[win] + np.polyval(np.polyfit(tl[win], msd[args.mobile][win], 1), 0),
                   color=INK2, lw=1, ls="--", label=f"fit: D* = {D_tr:.2e} cm²/s")
        ax[0].set(xlabel="lag time (ps)", ylabel="MSD (Å²)", title="Mean squared displacement")
        ax[0].legend(frameon=False, fontsize=8)
        ax[1].plot(log["time_ps"], T, color=SERIES[0], lw=1)
        ax[1].axhline(Tset, color=INK2, lw=1, ls="--")
        ax[1].set(xlabel="time (ps)", ylabel="T (K)", title="Temperature")
        ax[2].plot(tE, E / N, color=SERIES[0], lw=1)
        ax[2].set(xlabel="time (ps)", ylabel="E_pot (eV/atom)", title="Potential energy")
        for a in ax:
            a.grid(color=GRID, lw=0.6)
            a.title.set_color(INK)
        fig.suptitle(os.path.basename(rd), color=INK, fontsize=10)
        fig.savefig(os.path.join(rd, "check.png"), dpi=150)
        plt.close(fig)

# ---------- summary table ----------
if rows:
    hdr = f"{'run':<22}{'t,ps':>7}{'T±σ (exp)':>16}{'Edrift':>9}{'D_Li*':>11}{'D_σ':>11}{'H_R~':>6}{'moved':>7}  {'frame':<7}framework MSD plateau / slope"
    print(hdr)
    for r in rows:
        fwtxt = ", ".join(f"{k} {r['framework_msd_plateau_A2'][k]:.2f}/{r['framework_msd_slope_A2_ps'][k]:+.4f}"
                          for k in r["framework_msd_plateau_A2"])
        print(f"{r['run']:<22}{r['t_ps']:>7.0f}{r['T_mean']:>8.0f}±{r['T_std']:<3.0f}({r['T_std_expected']:.0f}){r['Epot_drift_eV_atom_ns']:>9.3f}"
              f"{r['D_tracer_cm2_s']:>11.2e}{r['D_sigma_cm2_s']:>11.2e}{(format(r['Haven_ratio_rough'], '.2f') if r['Haven_ratio_rough'] is not None else '—'):>6}{r['frac_Li_moved_gt_2A']:>7.2f}  {r['framework_status']:<7}{fwtxt}")
    if len(rows) >= 2:
        Ts = np.array([json.load(open(os.path.join(rd.rstrip('/'), 'info.json'))).get("T", 0) for rd in args.runs
                       if os.path.exists(os.path.join(rd.rstrip('/'), 'check.json'))], float)
        Ds = np.array([r["D_tracer_cm2_s"] if r["framework_status"] == "OK" else np.nan for r in rows])
        keep = np.isfinite(Ds)
        if keep.sum() < len(Ds):
            print(f"\nArrhenius uses only runs with framework OK: {keep.sum()} of {len(Ds)}")
        Ts, Ds = Ts[keep], Ds[keep]
        if len(Ds) >= 2 and np.all(Ds > 0) and np.all(Ts > 0):
            k = 8.617333e-5
            sl, ic = np.polyfit(1 / (k * Ts), np.log(Ds), 1)
            print(f"\nArrhenius (unweighted, rough): Ea = {-sl:.3f} eV, D0 = {np.exp(ic):.2e} cm²/s"
                  f"  -> D(300 K) ≈ {np.exp(ic + sl / (k * 300)):.2e} cm²/s")
