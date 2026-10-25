"""Single-panel figures with large fonts for the talk -> ../09_presentation/figures.

Run:  python make_slide_figures.py      (needs numpy, matplotlib, ase)
Same data as make_paper_figures.py; background matches the slides (#FAF9F6).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "09_presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NEB_A = ROOT / "03_data_NEB" / "A_idealized_c_channel_NEB"
EV = ROOT / "02_data_concerted_events"
MD = ROOT / "04_data_MD_run"

BG = "#FAF9F6"
INK = "#14213D"
C = ["#2a6fdb", "#e8743b", "#19a979", "#945ecf", "#6b7280"]
TAU_FRAMES = 47

plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 18, "legend.fontsize": 15, "xtick.labelsize": 15,
    "ytick.labelsize": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2.4,
})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def profile(path):
    rows = read_csv(path)
    return (np.array([float(r["path_A"]) for r in rows]),
            np.array([float(r["E_minus_E0_eV"]) for r in rows]))


def neb():
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for i, n in enumerate([1, 2, 3, 4]):
        x, e = profile(NEB_A / f"profile_n{n}_ions_moving.csv")
        lab = "1 ion (no stable end state)" if n == 1 else f"{n} ions"
        ax.plot(x, e, "o" + ("--" if n == 1 else "-"), color=C[i], ms=6, label=lab)
    ax.axhline(0.20, ls=":", c=C[4], lw=2)
    ax.text(0.2, 0.212, "DFT, concerted: 0.20 eV", color=C[4], fontsize=15)
    ax.annotate("0.15 eV", xy=(5.86, 0.152), xytext=(7.6, 0.17), color=C[3], fontsize=16,
                arrowprops=dict(arrowstyle="->", color=C[3]))
    ax.set(xlabel="path length (Å)", ylabel="E − E(initial) (eV)", ylim=(-0.2, 0.26))
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "neb.png")
    plt.close(fig)


def load_unwrapped():
    traj = read(MD / "md_trajectory_300ps_LINK.traj", index=":")
    li = np.array([s == "Li" for s in traj[0].get_chemical_symbols()])
    cell = traj[0].cell.array
    pos = np.array([a.get_positions() for a in traj])
    d = np.diff(pos, axis=0) @ np.linalg.inv(cell)
    d -= np.round(d)
    d = d @ cell
    d -= d[:, ~li].mean(axis=1, keepdims=True)
    return np.concatenate([pos[:1], pos[0] + np.cumsum(d, axis=0)]), li


def displacement(un, li):
    starts = np.arange(0, len(un) - TAU_FRAMES, TAU_FRAMES)
    dr = np.linalg.norm(un[starts + TAU_FRAMES][:, li] - un[starts][:, li], axis=-1).ravel()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.hist(dr, bins=np.arange(0, 6.05, 0.1), color=C[0])
    ax.axvline(1.8, ls="--", c=INK, lw=2)
    ax.text(1.9, ax.get_ylim()[1] * 0.9, "hop threshold\n1.8 Å", fontsize=15)
    ax.text(0.42, ax.get_ylim()[1] * 0.3, "vibration\nin a site", fontsize=15, color=BG)
    ax.text(3.2, ax.get_ylim()[1] * 0.35, "hops", fontsize=15, color=C[4])
    ax.set(xlabel="Li displacement in τ = 2.35 ps (Å)", ylabel="count")
    fig.savefig(OUT / "displacement.png")
    plt.close(fig)


def size_distribution():
    rows = read_csv(EV / "SevenNet_700K_event_size_histogram.csv")
    n = np.array([int(r["n"]) for r in rows])
    p = np.array([float(r["P_ion_weighted"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(n, p, color=[C[4]] + [C[1]] * (len(n) - 1))
    ax.text(1, p[0] + 0.006, "single\nhops", ha="center", fontsize=14, color=C[4])
    ax.text(7.5, 0.13, "concerted, n ≥ 2:\n83 % of all hops", fontsize=16, color=C[1])
    ax.set(xlabel="ions per event, n", ylabel="P(n), ion-weighted", xticks=range(1, 14))
    fig.savefig(OUT / "size_distribution.png")
    plt.close(fig)


def msd(un, li):
    dt = 0.05
    lags = np.unique(np.linspace(1, len(un) // 2, 100).astype(int))
    dsp = [un[l:, li] - un[:-l, li] for l in lags]
    t = lags * dt
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(t, [np.mean(np.sum(d ** 2, -1)) for d in dsp], c=C[0], label="total")
    ax.plot(t, [np.mean(d[..., 2] ** 2) for d in dsp], c=C[1], label="along c (1 direction)")
    ax.plot(t, [np.mean(np.sum(d[..., :2] ** 2, -1)) for d in dsp], c=C[2], label="ab plane (2 directions)")
    ax.set(xlabel="lag time (ps)", ylabel="Li MSD (Å²)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "msd.png")
    plt.close(fig)


def scan():
    rows = read_csv(EV / "SevenNet_700K_parameter_scan_27_combinations.csv")
    col = {"1.4": C[1], "1.8": C[0], "2.2": C[2]}
    mk = {"1.6": "v", "2.0": "o", "2.4": "^"}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in rows:
        main = r["d_hop_A"] == "1.8" and r["tau_ps"] == "2.35" and r["r_cut_A"] == "2.0"
        ax.scatter(float(r["mean_n_ion_weighted"]), float(r["concerted_events_per_ps"]),
                   marker=mk[r["r_cut_A"]], color=col[r["d_hop_A"]], s=200 if main else 70,
                   edgecolor=INK if main else "none", lw=2)
    for dh, cc in col.items():
        ax.scatter([], [], color=cc, marker="s", s=70, label=f"d_hop = {dh} Å")
    for rc, mm in mk.items():
        ax.scatter([], [], color=C[4], marker=mm, s=70, label=f"r_cut = {rc} Å")
    ax.annotate("main", xy=(3.9, 3.03), xytext=(5.6, 3.9), fontsize=15,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set(xlabel="mean event size ⟨n⟩", ylabel="concerted events per ps", ylim=(1.5, 5.9))
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=13, columnspacing=0.6)
    fig.savefig(OUT / "scan.png")
    plt.close(fig)


if __name__ == "__main__":
    neb()
    size_distribution()
    scan()
    un, li = load_unwrapped()
    displacement(un, li)
    msd(un, li)
