"""Figures for the final report (>=300 dpi, 10 pt fonts at text width) -> ../08_report_LaTeX/figures.

Run:  python make_paper_figures.py      (needs numpy, matplotlib, ase)
Also prints the numbers quoted in the report (N_eff and RSD of D after He et al. 2018).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "08_report_LaTeX" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NEB_A = ROOT / "03_data_NEB" / "A_idealized_c_channel_NEB"
NEB_B = ROOT / "03_data_NEB" / "B_NEB_on_events_from_MD"
EV = ROOT / "02_data_concerted_events"
MD = ROOT / "04_data_MD_run"

DFT_CONCERTED = 0.20   # He et al. 2017, NEB, LGPS
TAU_FRAMES = 47        # 2.35 ps / 0.05 ps
TEXTWIDTH = 6.5        # inch

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 330, "savefig.bbox": "tight",
})
C = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf", "#555555"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def profile(path):
    rows = read_csv(path)
    return (np.array([float(r["path_A"]) for r in rows]),
            np.array([float(r["E_minus_E0_eV"]) for r in rows]))


def label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=11)


def fig_neb():
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.1))
    summ = json.load(open(NEB_A / "summary_barriers.json"))["runs"]
    for i, n in enumerate([1, 2, 3, 4]):
        x, e = profile(NEB_A / f"profile_n{n}_ions_moving.csv")
        ls = "--" if n == 1 else "-"
        lab = "n = 1 (end point relaxed back)" if n == 1 else f"n = {n} ions"
        ax[0].plot(x, e, "o" + ls, color=C[i], ms=3, lw=1.2, label=lab)
    ax[0].axhline(DFT_CONCERTED, ls=":", c=C[4], lw=1)
    ax[0].text(0.3, DFT_CONCERTED + 0.012, "DFT concerted, 0.20 eV [He 2017]", color=C[4], fontsize=9)
    ax[0].set(xlabel="path length (Å)", ylabel=r"$E - E_{\rm initial}$ (eV)", ylim=(-0.2, 0.27))
    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2, fontsize=9)
    label(ax[0], "a")

    for i, c in enumerate([8, 11]):
        x, e = profile(NEB_B / f"profile_event_cand{c}.csv")
        ax[1].plot(x, e, "o-", color=C[i], ms=3, lw=1.2, label=f"hop at {['169.2', '235.0'][i]} ps")
    ax[1].axhline(DFT_CONCERTED, ls=":", c=C[4], lw=1)
    ax[1].set(xlabel="path length (Å)", ylabel=r"$E - E_{\rm initial}$ (eV)", ylim=(-0.2, 0.27))
    ax[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2)
    label(ax[1], "b")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig1_neb.png")
    plt.close(fig)


def load_unwrapped():
    traj = read(MD / "md_trajectory_300ps_LINK.traj", index=":")
    li = np.array([s == "Li" for s in traj[0].get_chemical_symbols()])
    cell = traj[0].cell.array
    inv = np.linalg.inv(cell)
    pos = np.array([a.get_positions() for a in traj])
    d = np.diff(pos, axis=0) @ inv
    d -= np.round(d)
    d = d @ cell
    d -= d[:, ~li].mean(axis=1, keepdims=True)     # remove framework drift
    un = np.concatenate([pos[:1], pos[0] + np.cumsum(d, axis=0)])
    return un, li


def fig_events(un, li):
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    starts = np.arange(0, len(un) - TAU_FRAMES, TAU_FRAMES)
    dr = np.linalg.norm(un[starts + TAU_FRAMES][:, li] - un[starts][:, li], axis=-1).ravel()
    ax[0].hist(dr, bins=np.arange(0, 6.05, 0.1), color=C[0])
    top = ax[0].get_ylim()[1]
    for x, h in [(1.4, 0.95), (1.8, 0.83), (2.2, 0.71)]:
        ax[0].axvline(x, ls="--" if x == 1.8 else ":", c="k", lw=0.9)
        ax[0].text(x + 0.05, top * 0.97, f"{x} Å", fontsize=8.5, rotation=90, va="top")
    ax[0].set(xlabel=r"Li displacement within $\tau$ = 2.35 ps (Å)", ylabel="count")
    label(ax[0], "a")

    def hist(p):
        rows = read_csv(p)
        return np.array([int(r["n"]) for r in rows]), np.array([float(r["P_ion_weighted"]) for r in rows])

    n1, p1 = hist(EV / "SevenNet_700K_event_size_histogram.csv")
    n2, p2 = hist(EV / "SevenNet_700K_event_size_histogram_Sato_parameters_dhop1.4.csv")
    ax[1].bar(n1 - 0.2, p1, 0.4, color=C[0], label=r"$d_{\rm hop}$ = 1.8 Å (main)")
    ax[1].bar(n2 + 0.2, p2, 0.4, color=C[1], label=r"$d_{\rm hop}$ = 1.4 Å (as in [Sato 2026])")
    ax[1].set(xlabel="ions per event, n", ylabel=r"$P(n)$, ion-weighted", xticks=range(1, 14))
    ax[1].legend(frameon=False)
    label(ax[1], "b")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig2_events.png")
    plt.close(fig)


def fig_msd_scan(un, li):
    dt = 0.05
    lags = np.unique(np.linspace(1, len(un) - 1, 120).astype(int))
    dsp = [un[l:, li] - un[:-l, li] for l in lags]
    msd = np.array([np.mean(np.sum(d ** 2, axis=-1)) for d in dsp])
    msd_z = np.array([np.mean(d[..., 2] ** 2) for d in dsp])
    msd_xy = np.array([np.mean(np.sum(d[..., :2] ** 2, axis=-1)) for d in dsp])
    t = lags * dt
    fit = (t > 10) & (t < 150)
    slope = np.polyfit(t[fit], msd[fit], 1)[0]
    D = slope / 6 * 1e-4
    D_c = np.polyfit(t[fit], msd_z[fit], 1)[0] / 2 * 1e-4
    D_ab = np.polyfit(t[fit], msd_xy[fit], 1)[0] / 4 * 1e-4
    n_li = int(li.sum())
    n_eff = (msd * n_li).max() / 2.8 ** 2           # He et al. 2018, eq. 9, a = 2.8 A for LGPS
    rsd = 3.43 / np.sqrt(n_eff) + 0.04               # eq. 8

    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
    m = t <= 150
    ax[0].plot(t[m], msd[m], c=C[0], label="total")
    ax[0].plot(t[m], msd_z[m], c=C[1], label="along c")
    ax[0].plot(t[m], msd_xy[m], c=C[2], label="in ab plane")
    ax[0].set(xlabel="lag time (ps)", ylabel=r"Li MSD (Å$^2$)")
    ax[0].legend(frameon=False)
    label(ax[0], "a")

    rows = read_csv(EV / "SevenNet_700K_parameter_scan_27_combinations.csv")
    mk = {"1.6": "v", "2.0": "o", "2.4": "^"}
    col = {"1.4": C[1], "1.8": C[0], "2.2": C[2]}
    for r in rows:
        ax[1].scatter(float(r["mean_n_ion_weighted"]), float(r["concerted_events_per_ps"]),
                      marker=mk[r["r_cut_A"]], color=col[r["d_hop_A"]], s=22,
                      edgecolor="k" if r["tau_ps"] == "2.35" else "none", lw=0.6)
    for dh, cc in col.items():
        ax[1].scatter([], [], color=cc, marker="s", label=rf"$d_{{\rm hop}}$ = {dh} Å")
    for rc, mm in mk.items():
        ax[1].scatter([], [], color="grey", marker=mm, label=rf"$r_{{\rm cut}}$ = {rc} Å")
    ax[1].set(xlabel=r"mean event size $\langle n \rangle$ (ion-weighted)", ylabel=r"concerted events ps$^{-1}$")
    ax[1].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right", columnspacing=0.8, handletextpad=0.2)
    ax[1].set_ylim(1.5, 7.0)
    label(ax[1], "b")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig3_msd_scan.png")
    plt.close(fig)
    return D, D_c, D_ab, n_eff, rsd


def fig_appendix():
    log = read_csv(MD / "log.csv")
    t = np.array([float(r["time_ps"]) for r in log])
    T = np.array([float(r["T_inst_K"]) for r in log])
    E = np.array([float(r["Epot_eV"]) for r in log]) / 200
    fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))
    ax[0].plot(t, T, lw=0.3, c=C[0]); ax[0].axhline(700, c="k", ls="--", lw=0.8)
    ax[0].set(xlabel="time (ps)", ylabel="T (K)")
    ax[1].plot(t, E, lw=0.3, c=C[0])
    ax[1].set(xlabel="time (ps)", ylabel=r"$E_{\rm pot}$ (eV/atom)")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "figA1_md_stability.png")
    plt.close(fig)
    return T.mean(), T.std()


if __name__ == "__main__":
    fig_neb()
    un, li = load_unwrapped()
    fig_events(un, li)
    D, D_c, D_ab, n_eff, rsd = fig_msd_scan(un, li)
    Tm, Ts = fig_appendix()
    print(f"D* = {D:.3e} cm2/s; D_c = {D_c:.3e}; D_ab = {D_ab:.3e}; N_eff = {n_eff:.0f}; RSD(D) = {rsd:.3f}")
    print(f"<T> = {Tm:.1f} +- {Ts:.1f} K")
