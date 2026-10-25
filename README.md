# Concerted Li-ion migration in LGPS with a universal equivariant potential

Reproducibility package for the final project report (Introduction to Materials Informatics, Skoltech, October 2026):
**"Concerted Li-ion migration in Li₁₀GeP₂S₁₂ with a universal equivariant potential: PES validation and MD statistics of concerted events"**.

The report itself: [`08_report_LaTeX/report.pdf`](08_report_LaTeX/report.pdf) (LaTeX source: `report.tex`).

## Setup in one paragraph

Structure: ordered tetragonal LGPS, Materials Project entry mp-696128 (`05_structure/LGPS.cif`, 50 atoms, P4₂mc).
Potential: pretrained SevenNet-0 (`7net-0`, checkpoint `SevenNet_0__11Jul2024`, package `sevenn` 0.11.2), no fine-tuning, ASE 3.26, PyTorch 2.5.1, CPU.
MD: 2×1×2 supercell (200 atoms, 80 Li), NVT, Bussi thermostat (τ_T = 100 fs), 700 K, dt = 2 fs, seed 42, 10 ps equilibration + 300 ps production, frames every 50 fs.
NEB: climbing-image NEB, 7 images, IDPP, FIRE to 0.10 then CI to 0.05 eV/Å.
Concerted events: directed-graph model of Sato et al., Chem. Mater. 38, 287 (2026); main parameters d_hop = 1.8 Å, τ = 2.35 ps, r_cut = 2.0 Å; 27-combination robustness scan.
Exact package versions: `06_scripts_methods/environment-sevennet.txt`.

## Contents

| Folder | Content |
|---|---|
| `01_figures/` | Figures (PNG) |
| `02_data_concerted_events/` | Event lists, size histograms, summaries (main and Sato parameters), 27-combination scan, Haven ratio |
| `03_data_NEB/A_idealized_c_channel_NEB/` | NEB for n = 1–4 ions in the ideal c-channel: profiles (`.csv`), paths (`.traj`), logs, `summary_barriers.json` |
| `03_data_NEB/B_NEB_on_events_from_MD/` | NEB on hops extracted from MD (2 converged single-ion hops), candidate logs for n = 2–4 |
| `04_data_MD_run/` | 300 ps trajectory (`md_trajectory_300ps_LINK.traj`, ASE format, 6001 frames), `log.csv`, settings, time-step test |
| `05_structure/` | Input structure and its Materials Project record |
| `06_scripts_methods/` | All scripts (see below) |
| `07_supplementary_NOT_main_text/` | Not used in the main text: MACE-MP-0 comparison, unconverged NEB attempts (do not use) |
| `08_report_LaTeX/` | Report source, figures and PDF |
| `README_ru.md` | Detailed file description in Russian |

## Reproducing the results

```bash
cd 06_scripts_methods
# 1. MD (about 20 h on 8 CPU cores); writes runs/LGPS_sevennet_700K/
python run_md.py --potential sevennet --material LGPS --T 700
# 2. NEB in the ideal c-channel (set A)
python neb_lgps_sevennet.py --calc sevennet --nmax 4 --nimages 7
# 3. NEB on events extracted from MD (set B): see neb_from_md_AS_USED_run.sbatch for the exact call
# 4. Concerted events + robustness scan
python concerted_events.py runs/LGPS_sevennet_700K --d_hop 1.8 1.4 2.2 --tau 2.35 1.5 4 --rcut 2.0 1.6 2.4
# 5. Haven ratio from the trajectory
python haven_ratio.py ../04_data_MD_run/md_trajectory_300ps_LINK.traj
# 6. Report figures and the D*/RSD numbers quoted in the report
python make_paper_figures.py
```

The `*.sbatch` files are the SLURM scripts used on the cluster; `*_AS_USED*` files are the exact versions that produced the data.
