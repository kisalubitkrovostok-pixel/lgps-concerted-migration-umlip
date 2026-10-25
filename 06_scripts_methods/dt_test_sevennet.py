"""Timestep check: energy drift in NVE with dt = 1 and 2 fs, starting from the same equilibrated state.

  python dt_test.py --potential mace --material LGPS --T 1100

Worst case = highest temperature. Criterion: |drift| of total energy well below ~1 meV/atom/ps,
and dt = 2 fs close to dt = 1 fs. Result -> dt_test_<material>_<potential>_<T>K.json
"""
import argparse, json, os, sys, time
import numpy as np
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

sys.argv, _argv = [sys.argv[0]], sys.argv           # reuse calculator/structure setup from run_md.py
p = argparse.ArgumentParser()
p.add_argument("--potential", default="mace")
p.add_argument("--material", default="LGPS")
p.add_argument("--T", type=float, default=1100)
p.add_argument("--nve_ps", type=float, default=5.0)
p.add_argument("--equil_ps", type=float, default=2.0)
p.add_argument("--structure", default=None)
p.add_argument("--supercell", type=int, nargs=3, default=None)
args = p.parse_args(_argv[1:])

STRUCT = {"LGPS": "structures/LGPS.cif", "LLZO": "structures/llzo_cub_ga.POSCAR"}
SUPERCELL = {"LGPS": (2, 2, 2), "LLZO": (1, 1, 1)}
nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


def make_calc(name):
    if name == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.0, epsilon=0.01, rc=5.0, smooth=True)
    import torch
    torch.set_num_threads(nthreads)
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device="cpu", default_dtype="float32")
    if name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator("7net-0", device="cpu")
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device="cpu")
    if name == "m3gnet":
        import matgl
        from matgl.ext.ase import PESCalculator
        return PESCalculator(matgl.load_model("M3GNet-MP-2021.2.8-PES"))


calc = make_calc(args.potential)
sc = tuple(args.supercell) if args.supercell else SUPERCELL[args.material]
atoms = read("runs/LGPS_sevennet_700K/md.traj", index=-1)
if len(atoms) != 200 or not atoms.has("momenta"):
    raise SystemExit("Ожидались 200 атомов и сохранённые импульсы")
print("Старт из SevenNet MD:", len(atoms), "атомов;",
      "T =", atoms.get_temperature(), "K", flush=True)
atoms.pbc = True
atoms.calc = calc
n = len(atoms)

start = atoms.copy()

res = {"source_trajectory": "runs/LGPS_sevennet_700K/md.traj", "source_frame": -1, "material": args.material, "potential": args.potential, "T": args.T, "natoms": n}
for dt in (1.0, 2.0):
    a = start.copy()
    a.calc = calc
    nsteps = int(args.nve_ps * 1000 / dt)
    t_ps, etot = [], []
    dyn = VelocityVerlet(a, dt * units.fs)

    def rec():
        t_ps.append(dyn.nsteps * dt / 1000)
        etot.append(a.get_potential_energy() + a.get_kinetic_energy())

    dyn.attach(rec, interval=max(1, int(10 / dt)))   # every 10 fs
    t0 = time.perf_counter()
    dyn.run(nsteps)
    sps = (time.perf_counter() - t0) / nsteps
    slope = np.polyfit(t_ps, etot, 1)[0]              # eV/ps
    e = np.array(etot)
    res[f"dt{int(dt)}fs"] = {
        "drift_meV_per_atom_per_ps": round(float(1000 * slope / n), 4),
        "fluct_std_meV_per_atom": round(1000 * float(np.std(e - np.polyval(np.polyfit(t_ps, e, 1), t_ps))) / n, 4),
        "sec_per_step": round(sps, 4),
        "T_final_K": round(float(a.get_temperature()), 0),
    }
    print(f"dt = {dt} fs: {res[f'dt{int(dt)}fs']}", flush=True)

out = os.path.join(os.environ["CHECK_OUT"], "dt_test.json")
json.dump(res, open(out, "w"), indent=1)
print("->", out)
