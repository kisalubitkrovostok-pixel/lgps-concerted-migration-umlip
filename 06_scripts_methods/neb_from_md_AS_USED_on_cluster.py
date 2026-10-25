"""NEB on concerted-migration events taken directly from the MD trajectory.

For events of size n found by concerted_events.py (runs/<run>/concerted/paths.csv):
  A = quench(snapshot at t)                       all atoms relaxed
  B = quench(snapshot at t with ONLY the event ions moved to their positions at t + tau)
  checks: every event ion moved > --min_hop A between A and B, no other Li moved > --max_other A
  CI-NEB A -> B; barrier metrics:
     E_fwd = E_max - E_A,  E_bwd = E_max - E_B,  E_KRA = E_max - (E_A + E_B)/2   (kinetically resolved barrier)

  python neb_from_md.py runs/LGPS_sevennet_700K --potential sevennet --n 1 --per_n 4
  (run one job per n in parallel: --n 1, --n 2, --n 3, --n 4)

Outputs runs/<run>/neb_md/n<k>_e<j>/ : path.traj, profile.csv ; runs/<run>/neb_md/results_n<k>.csv
"""
import argparse, csv, json, os
import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
try:
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--potential", default="sevennet", choices=["mace", "sevennet", "chgnet", "m3gnet", "lj"])
p.add_argument("--n", type=int, nargs="+", default=[1, 2, 3, 4])
p.add_argument("--per_n", type=int, default=4, help="successful NEBs per event size")
p.add_argument("--max_tries", type=int, default=15, help="candidates to try per n")
p.add_argument("--nimages", type=int, default=7)
p.add_argument("--fmax", type=float, default=0.05)
p.add_argument("--min_hop", type=float, default=1.0)
p.add_argument("--max_other", type=float, default=0.7)
p.add_argument("--max_dR", type=float, default=12.0, help="skip events with longer apparent displacement")
args = p.parse_args()

nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


def make_calc(name):
    if name == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.0, epsilon=0.01, rc=5.0, smooth=True)
    import torch
    torch.set_num_threads(nthreads)
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device="cpu", default_dtype="float64")
    if name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator("7net-0", device="cpu")
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device="cpu")
    import matgl
    from matgl.ext.ase import PESCalculator
    return PESCalculator(matgl.load_model("M3GNet-MP-2021.2.8-PES"))


calc = make_calc(args.potential)
rd = args.run.rstrip("/")
summ = json.load(open(os.path.join(os.environ["NEB_RESULTS"], "events", "summary.json")))
tau = float(summ["tau_ps"])
rows = list(csv.DictReader(open(os.path.join(os.environ["NEB_RESULTS"], "events", "paths.csv"))))
tr = Trajectory(os.path.join(rd, "md.traj"))
a0 = tr[0]
sym = np.array(a0.get_chemical_symbols())
li_global = np.flatnonzero(sym == "Li")                     # paths.csv indices are within the Li sub-array
log = np.genfromtxt(os.path.join(rd, "log.csv"), delimiter=",", names=True)
dtf = float(log["time_ps"][1] - log["time_ps"][0])
L = int(round(tau / dtf))
out = os.environ["NEB_RESULTS"]
os.makedirs(out, exist_ok=True)
print(f"{rd}: tau = {tau} ps ({L} frames), {len(rows)} events in paths.csv", flush=True)


def quench(at):
    at = at.copy()
    at.set_constraint()
    at.calc = calc
    if not BFGS(at, logfile="-").run(fmax=0.02, steps=800):
        return None
    return at


def mic_norm(v, cell):
    return np.linalg.norm(find_mic(v, cell, pbc=True)[0], axis=-1)


for n in args.n:
    cand = [r for r in rows if int(r["n_ions"]) == n and int(r["closed_loop"]) == 0
            and float(r["dR_A"]) <= args.max_dR]
    if not cand:
        print(f"n = {n}: no events", flush=True)
        continue
    # spread candidates over the whole trajectory
    idx = np.unique(np.linspace(0, len(cand) - 1, min(len(cand), args.max_tries)).astype(int))
    cand = [cand[i] for i in idx]
    results, done = [], 0
    for ci, r in enumerate(cand):
        if done >= args.per_n:
            break
        f0 = int(round(float(r["t_start_ps"]) / dtf))
        f1 = f0 + L
        if f1 >= len(tr):
            continue
        ions = [int(li_global[int(k)]) for k in r["li_indices"].split()]
        s0, s1 = tr[f0], tr[f1]
        A0 = s0.copy()
        B0 = s0.copy()
        B0.positions[ions] = s0.positions[ions] + (s1.positions[ions] - s0.positions[ions])   # unwrapped displacement
        A, B = quench(A0), quench(B0)
        if A is None or B is None:
            print("Концы не сошлись — пропускаю кандидат", flush=True)
            continue
        d_ev = mic_norm(B.positions[ions] - A.positions[ions], A.cell)
        others = np.setdiff1d(li_global, ions)
        d_ot = mic_norm(B.positions[others] - A.positions[others], A.cell)
        ok = bool(np.all(d_ev > args.min_hop) and d_ot.max() < args.max_other)
        tag = f"n{n}_cand{ci}"
        print(f"n={n} cand {ci} t={r['t_start_ps']} ps ions {ions}: event shifts {np.round(d_ev, 2)}, "
              f"max other Li {d_ot.max():.2f} -> {'OK' if ok else 'skip'}", flush=True)
        if not ok:
            continue
        images = [A] + [A.copy() for _ in range(args.nimages)] + [B]
        for im in images:
            im.calc = calc
        neb = NEB(images, climb=False, method="improvedtangent", allow_shared_calculator=True)
        neb.interpolate(method="idpp", mic=True)
        d = os.path.join(out, tag)
        os.makedirs(d, exist_ok=True)
        opt = FIRE(neb, logfile=os.path.join(d, "neb.log"))
        opt.run(fmax=0.10, steps=400)
        neb.climb = True
        conv = bool(opt.run(fmax=args.fmax, steps=800))
        E = np.array([im.get_potential_energy() for im in images])
        Emax = E.max()
        res = {"n": n, "event": done, "t_start_ps": float(r["t_start_ps"]), "ions": " ".join(map(str, ions)),
               "event_shifts_A": " ".join(f"{x:.2f}" for x in d_ev), "dR_md_A": float(r["dR_A"]),
               "E_fwd_eV": round(float(Emax - E[0]), 4), "E_bwd_eV": round(float(Emax - E[-1]), 4),
               "E_KRA_eV": round(float(Emax - 0.5 * (E[0] + E[-1])), 4), "dE_eV": round(float(E[-1] - E[0]), 4),
               "converged": conv}
        s = [0.0]
        for a_, b_ in zip(images[:-1], images[1:]):
            s.append(s[-1] + float(np.linalg.norm(find_mic(b_.positions - a_.positions, a_.cell)[0])))
        write(os.path.join(d, "path.traj"), images)
        with open(os.path.join(d, "profile.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "path_A", "E_minus_E0_eV"])
            for i, (si, ei) in enumerate(zip(s, E - E[0])):
                w.writerow([i, round(si, 4), round(float(ei), 5)])
        print("   ", json.dumps(res), flush=True)
        results.append(res)
        with open(os.path.join(out, f"results_n{n}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        if conv:
            done += 1
    if results:
        with open(os.path.join(out, f"results_n{n}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        kra = [x["E_KRA_eV"] for x in results if x["converged"]]
        print(f"n = {n}: {len(results)} NEBs, converged {len(kra)}; E_KRA median "
              f"{np.median(kra) if kra else float('nan'):.3f} eV, range "
              f"{(min(kra), max(kra)) if kra else '-'}", flush=True)
