"""Haven ratio H_R = D*/D_sigma from one MD trajectory (full run + three 100-ps blocks).

  python haven_ratio.py ../04_data_MD_run/md_trajectory_300ps_LINK.traj

D_sigma = N_Li * D_com (Li centre-of-mass diffusivity); very noisy from a single trajectory, hence the blocks.
Result for SevenNet 700 K: H_R = 0.41 (full), blocks 0.62 / 0.54 / 0.23."""
import numpy as np, sys
from ase.io import read
traj = read(sys.argv[1], index=":")
li = np.array([s == "Li" for s in traj[0].get_chemical_symbols()]); N = li.sum()
cell = traj[0].cell.array; inv = np.linalg.inv(cell)
pos = np.array([a.get_positions() for a in traj])
d = np.diff(pos, axis=0) @ inv; d -= np.round(d); d = d @ cell
d -= d[:, ~li].mean(axis=1, keepdims=True)
un = np.concatenate([pos[:1], pos[0] + np.cumsum(d, axis=0)])
dt = 0.05
def fit(x, tmax):
    lags = np.arange(20, int(tmax / dt), 20); t = lags * dt
    ms = np.array([np.mean(np.sum((x[l:] - x[:-l]) ** 2, axis=-1)) for l in lags])
    m = (t > 5) & (t < tmax)
    return np.polyfit(t[m], ms[m], 1)[0] / 6 * 1e-4
def haven(seg, tmax):
    Dt = fit(seg[:, li], tmax)                       # per-ion tracer
    com = seg[:, li].mean(axis=1)
    Ds = fit(com, tmax) * N                          # N * D_com
    return Dt, Ds, Dt / Ds
Dt, Ds, H = haven(un, 100)
print(f"full 300 ps (fit 5-100 ps): D*={Dt:.3e}  D_sigma={Ds:.3e}  H_R={H:.2f}")
for b in range(3):
    seg = un[b * 2000:(b + 1) * 2000 + 1]
    print(f"block {b} (100 ps, fit 5-33 ps):", "D*=%.2e D_sigma=%.2e H_R=%.2f" % haven(seg, 33))
