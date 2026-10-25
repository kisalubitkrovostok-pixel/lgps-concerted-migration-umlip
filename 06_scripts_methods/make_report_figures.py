"""Builds the report figures in ../01_figures from the data in this folder.

Run from anywhere:  python make_report_figures.py
Needs: numpy, matplotlib, ase.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "01_figures"
NEB_A = ROOT / "03_data_NEB" / "A_idealized_c_channel_NEB"
NEB_B = ROOT / "03_data_NEB" / "B_NEB_on_events_from_MD"
EV = ROOT / "02_data_concerted_events"
MD = ROOT / "04_data_MD_run"
SUP = ROOT / "07_supplementary_NOT_main_text"

# DFT reference values quoted in the proposal (He et al., Nat. Commun. 2017)
DFT_CONCERTED = 0.2
DFT_SINGLE = 0.5

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def profile(path):
    rows = read_csv(path)
    return (np.array([float(r["path_A"]) for r in rows]),
            np.array([float(r["E_minus_E0_eV"]) for r in rows]))


def fig1_neb_idealized():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, n in enumerate([1, 2, 3, 4]):
        x, e = profile(NEB_A / f"profile_n{n}_ions_moving.csv")
        label = "single-ion hop" if n == 1 else f"concerted, {n} ions"
        ax.plot(x / x[-1], e, "o-", color=COLORS[i], label=label, ms=4)
    ax.axhline(DFT_CONCERTED, ls="--", c="grey", lw=1)
    ax.text(0.01, DFT_CONCERTED + 0.01, "DFT concerted ≈ 0.2 eV (He et al. 2017)", color="grey", fontsize=9)
    ax.axhline(DFT_SINGLE, ls=":", c="grey", lw=1)
    ax.text(0.01, DFT_SINGLE + 0.01, "DFT single-ion ≈ 0.5 eV", color="grey", fontsize=9)
    ax.set_xlabel("normalized reaction coordinate")
    ax.set_ylabel("E − E(initial), eV")
    ax.set_title("SevenNet CI-NEB, Li migration along c-channel")
    ax.set_ylim(-0.2, 0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_NEB_SevenNet_single_vs_concerted_vs_DFT.png", dpi=200)
    plt.close(fig)


def fig2_neb_md_events():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for i, c in enumerate([8, 11]):
        x, e = profile(NEB_B / f"profile_event_cand{c}.csv")
        ax.plot(x, e, "o-", color=COLORS[i], ms=4, label=f"single-ion hop from MD (cand{c})")
    ax.axhline(DFT_CONCERTED, ls="--", c="grey", lw=1)
    ax.text(0.02, DFT_CONCERTED + 0.005, "DFT concerted ≈ 0.2 eV", color="grey", fontsize=9)
    ax.set_xlabel("path length, Å")
    ax.set_ylabel("E − E(initial), eV")
    ax.set_title("SevenNet CI-NEB on hops taken from 700 K MD")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_NEB_SevenNet_on_hops_extracted_from_MD.png", dpi=200)
    plt.close(fig)


def fig4_parameter_sensitivity():
    rows = read_csv(EV / "SevenNet_700K_parameter_scan_27_combinations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = [("concerted_events_per_ps", "concerted events (n ≥ 2) per ps"),
            ("frac_hops_in_concerted", "fraction of hops in concerted events")]
    for ax, (k, lab) in zip(axes, keys):
        for i, rc in enumerate(sorted({r["r_cut_A"] for r in rows})):
            sub = [r for r in rows if r["r_cut_A"] == rc]
            xs = [f'{r["d_hop_A"]}/{r["tau_ps"]}' for r in sub]
            ax.plot(range(len(sub)), [float(r[k]) for r in sub], "o-", color=COLORS[i], label=f"r_cut = {rc} Å")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=60, fontsize=8)
        ax.set_xlabel("d_hop (Å) / τ (ps)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False)
    fig.suptitle("Sensitivity of concerted-event statistics to detection parameters (SevenNet, 700 K, 300 ps)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_concerted_event_count_sensitivity_to_parameters.png", dpi=200)
    plt.close(fig)


def fig5_md_check_full():
    from ase.io import read

    log = read_csv(MD / "log.csv")
    t = np.array([float(r["time_ps"]) for r in log])
    T = np.array([float(r["T_inst_K"]) for r in log])
    E = np.array([float(r["Epot_eV"]) for r in log])
    info = json.load(open(MD / "info.json"))

    traj = read(MD / "md_trajectory_300ps_LINK.traj", index=":")
    li = np.array([a.symbol == "Li" for a in traj[0]])
    pos = np.array([a.get_positions() for a in traj])
    cell = traj[0].cell.array
    # unwrap with minimum image between consecutive frames
    inv = np.linalg.inv(cell)
    d = np.diff(pos, axis=0) @ inv
    d -= np.round(d)
    un = np.concatenate([pos[:1], pos[0] + np.cumsum(d @ cell, axis=0)])
    dt = t[1] - t[0]
    lags = np.arange(1, len(un) // 2, 10)
    msd = np.array([np.mean(np.sum((un[l:, li] - un[:-l, li]) ** 2, axis=-1)) for l in lags])
    msd_c = np.array([np.mean((un[l:, li, 2] - un[:-l, li, 2]) ** 2) for l in lags])
    lag_t = lags * dt
    fit = lag_t > 10
    slope = np.polyfit(lag_t[fit], msd[fit], 1)[0]
    D = slope / 6 * 1e-16 / 1e-12  # Å^2/ps -> cm^2/s

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(lag_t, msd, c=COLORS[0], label="Li total")
    axes[0].plot(lag_t, 3 * msd_c, c=COLORS[1], label="Li, 3·MSD_z (along c)")
    axes[0].plot(lag_t, slope * lag_t + np.polyval(np.polyfit(lag_t[fit], msd[fit], 1), 0) - 0, "k--", lw=1,
                 label=f"fit: D* = {D:.2e} cm²/s")
    axes[0].set_xlabel("lag time, ps"); axes[0].set_ylabel("MSD, Å²"); axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Li mean squared displacement")
    axes[1].plot(t, T, lw=0.5, c=COLORS[0]); axes[1].axhline(700, c="k", ls="--", lw=1)
    axes[1].set_xlabel("time, ps"); axes[1].set_ylabel("T, K"); axes[1].set_title("Temperature")
    axes[2].plot(t, E / info["natoms"], lw=0.5, c=COLORS[0])
    axes[2].set_xlabel("time, ps"); axes[2].set_ylabel("E_pot, eV/atom"); axes[2].set_title("Potential energy")
    fig.suptitle(f"LGPS, SevenNet, NVT 700 K, {t[-1]:.0f} ps, {info['natoms']} atoms")
    fig.tight_layout()
    fig.savefig(FIG / "figS2_SevenNet_700K_MD_check_full_300ps_MSD_T_Epot.png", dpi=200)
    plt.close(fig)
    return D


def figS3_sevennet_vs_mace():
    def hist(p):
        rows = read_csv(p)
        return np.array([int(r["n"]) for r in rows]), np.array([float(r["P_ion_weighted"]) for r in rows])

    n1, p1 = hist(EV / "SevenNet_700K_event_size_histogram.csv")
    n2, p2 = hist(SUP / "MACE_700K_comparison" / "MACE_700K_event_size_histogram.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(n1 - 0.2, p1, 0.4, color=COLORS[0], label="SevenNet (300 ps)")
    ax.bar(n2 + 0.2, p2, 0.4, color=COLORS[1], label="MACE-MP-0 (175 ps)")
    ax.set_xlabel("ions per event, n"); ax.set_ylabel("P(n), ion-weighted")
    ax.set_title("Event size distribution, LGPS 700 K")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figS3_event_size_SevenNet_vs_MACE_supplementary.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1_neb_idealized()
    fig2_neb_md_events()
    fig4_parameter_sensitivity()
    figS3_sevennet_vs_mace()
    print("D* from full trajectory, cm2/s:", fig5_md_check_full())
