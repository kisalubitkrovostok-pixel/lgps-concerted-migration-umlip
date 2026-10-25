"""Production NVT MD with a universal MLIP on CPU. Restartable: rerunning the same command continues.

  python run_md.py --prepare                         # check existing selected structures
  python run_md.py --potential mace --material LGPS --T 700

Defaults: 300 ps production, 10 ps equilibration (retained from supplied update),
LGPS 2x1x2 = 200 atoms, Bussi tau=100 fs, dt=2 fs, frames every 50 fs.
MACE loads local models/mace_mp0_medium.model; no download on compute nodes.
Restart uses the last production frame and saved RNG state. If interrupted during
initial equilibration, equilibration is repeated. Increase --ns to extend a run.
Run in a compute allocation, with the project directory as working directory.

Output: runs/<material>_<potential>_<T>K/
  md.traj    production frames every --every steps (unwrapped positions + momenta)
  log.csv    frame, time_ps, T_inst_K, Epot_eV
  info.json  settings, timing, ETA
"""
import argparse, os, json, time, csv, hashlib, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "models/matplotlib"))
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

p = argparse.ArgumentParser()
p.add_argument("--potential", choices=["mace", "sevennet", "chgnet", "m3gnet", "lj"])
p.add_argument("--material", choices=["LGPS", "LLZO"])
p.add_argument("--T", type=float)
p.add_argument("--ns", type=float, default=0.3, help="production length in ns (default: 300 ps)")
p.add_argument("--dt", type=float, default=2.0, help="fs")
p.add_argument("--every", type=int, default=25, help="save every N steps (25 x 2 fs = 50 fs)")
p.add_argument("--equil_ps", type=float, default=10.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--supercell", type=int, nargs=3, default=None)
p.add_argument("--structure", default=None, help="override structure file (testing)")
p.add_argument("--outdir", default="runs")
p.add_argument("--model", default=str(ROOT / "models/mace_mp0_medium.model"))
p.add_argument("--prepare", action="store_true")
args = p.parse_args()

STRUCT = {"LGPS": "structures/LGPS.cif", "LLZO": "structures/llzo_cub_ga.POSCAR"}
SUPERCELL = {"LGPS": (2, 1, 2), "LLZO": (1, 1, 1)}   # LGPS 200 atoms, LLZO 188 atoms


if args.prepare:
    if not os.path.exists(STRUCT["LGPS"]):
        raise SystemExit("Missing selected structures/LGPS.cif; no automatic structure substitution.")
    for m, f in STRUCT.items():
        if os.path.exists(f):
            a = read(f)
            print(f"{m}: {len(a)} atoms, cell {np.round(a.cell.lengths(), 2)}, "
                  f"supercell {SUPERCELL[m]} -> {len(a) * int(np.prod(SUPERCELL[m]))} atoms")
        else:
            print(f"{m}: MISSING {f}")
    raise SystemExit

if args.potential is None or args.material is None or args.T is None:
    p.error("--potential, --material and --T are required")
if not all(np.isfinite(x) for x in (args.T, args.ns, args.dt, args.equil_ps)):
    p.error("Parameters must be finite")
if min(args.T, args.ns, args.dt, args.every) <= 0 or args.equil_ps < 0:
    p.error("Invalid temperature, duration or interval")
if args.supercell and min(args.supercell) < 1:
    p.error("Supercell factors must be positive")
for duration, interval in ((args.ns * 1e6, args.dt * args.every),
                           (args.equil_ps * 1000, args.dt)):
    if not np.isclose(duration / interval, round(duration / interval), rtol=0, atol=1e-7):
        p.error("Duration must be an integer multiple of its step/save interval")

nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))


def make_calc(name):
    if name == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.0, epsilon=0.01, rc=5.0)
    import torch
    torch.set_num_threads(nthreads)
    if name == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=args.model, device="cpu", default_dtype="float32")
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


rng = np.random.default_rng(args.seed)

def make_dyn(atoms):
    from ase.md.bussi import Bussi
    return Bussi(atoms, args.dt * units.fs, temperature_K=args.T,
                 taut=100 * units.fs, rng=rng)


def pressure_GPa(atoms):
    """Virial (potential) pressure from the calculator stress, GPa; kinetic part not included."""
    try:
        return float(-atoms.get_stress(voigt=True)[:3].mean() / units.GPa)
    except Exception:
        return float("nan")


tag = f"{args.material}_{args.potential}_{args.T:g}K"
rd = os.path.join(args.outdir, tag)
os.makedirs(rd, exist_ok=True)
traj_path, log_path, info_path = (os.path.join(rd, x) for x in ("md.traj", "log.csv", "info.json"))
total_frames = int(round(args.ns * 1e6 / args.dt / args.every)) + 1
# Prevent concurrent writers to the same trajectory.
lock = open(os.path.join(rd, ".run.lock"), "a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Another job is writing this run directory")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

sfile = args.structure or STRUCT[args.material]
sc = tuple(args.supercell) if args.supercell else SUPERCELL[args.material]
settings = dict(material=args.material, potential=args.potential, T=args.T,
                dt_fs=args.dt, every=args.every, supercell=list(sc),
                equil_ps=args.equil_ps, seed=args.seed, thermostat="Bussi", taut_fs=100,
                structure_sha256=digest(sfile), dtype="float32")
if args.potential == "mace":
    if not Path(args.model).is_file():
        raise SystemExit(f"Local MACE weights missing: {args.model}")
    settings["model_sha256"] = digest(args.model)
settings_path = Path(rd) / "settings.json"
if settings_path.exists():
    old = json.loads(settings_path.read_text())
    if old != settings:
        raise SystemExit("Restart settings differ; use a separate --outdir")
else:
    if Path(traj_path).exists() and Path(traj_path).stat().st_size:
        raise SystemExit("Old trajectory has no settings.json; refusing unverified restart")
    settings_path.write_text(json.dumps(settings, indent=2))
calc = make_calc(args.potential)

# ---------- start or restart ----------
frames = 0
if os.path.exists(traj_path) and os.path.getsize(traj_path) > 0:
    with Trajectory(traj_path) as tr:
        frames = len(tr)
        if frames == 0:
            raise SystemExit("Empty trajectory; inspect run directory before restarting")
        atoms = tr[-1]
    if "rng_state" not in atoms.info:
        raise SystemExit("Checkpoint lacks RNG state")
    rng.bit_generator.state = atoms.info["rng_state"]
    # Rebuild CSV from committed frames if interrupted between trajectory and CSV writes.
    with Trajectory(traj_path) as tr, open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ps", "T_inst_K", "Epot_eV"])
        for i, a in enumerate(tr):
            w.writerow([i, i * args.every * args.dt / 1000,
                        a.info["T_inst_K"], a.info["Epot_eV"]])
    print(f"[{tag}] restart from frame {frames}/{total_frames}", flush=True)
else:
    sfile = args.structure or STRUCT[args.material]
    sc = tuple(args.supercell) if args.supercell else SUPERCELL[args.material]
    atoms = read(sfile).repeat(sc)
    atoms.pbc = True
    atoms.calc = calc
    P0 = pressure_GPa(atoms)
    Fmax0 = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    json.dump({"P_virial_start_GPa": round(P0, 3), "max_force_start_eV_A": round(Fmax0, 3),
               "volume_A3": round(atoms.get_volume(), 2), "seed": args.seed},
              open(os.path.join(rd, "start.json"), "w"), indent=1)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.T, rng=rng)
    Stationary(atoms)
    n_eq = int(args.equil_ps * 1000 / args.dt)
    print(f"[{tag}] {len(atoms)} atoms, supercell {sc}; P_virial(start) = {P0:.2f} GPa, "
          f"max|F| = {Fmax0:.2f} eV/A; equilibration {n_eq} steps", flush=True)
    t0 = time.perf_counter()
    eq_dyn = make_dyn(atoms)
    with open(os.path.join(rd, "equil.csv"), "w", newline="") as ef:
        ew = csv.writer(ef)
        ew.writerow(["step", "time_ps", "T_inst_K", "Epot_eV"])
        def eq_log():
            temp, energy = atoms.get_temperature(), atoms.get_potential_energy()
            if not np.isfinite([temp, energy]).all():
                raise RuntimeError("Non-finite state during equilibration")
            ew.writerow([eq_dyn.nsteps, eq_dyn.nsteps * args.dt / 1000, temp, energy])
            ef.flush()
            print(f"[{tag}] equil step={eq_dyn.nsteps}/{n_eq} T={temp:.1f} K", flush=True)
        eq_dyn.attach(eq_log, interval=250)
        eq_dyn.run(n_eq)
    print(f"[{tag}] equilibrated: T = {atoms.get_temperature():.0f} K, "
          f"{(time.perf_counter() - t0) / max(n_eq, 1):.3f} s/step", flush=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["frame", "time_ps", "T_inst_K", "Epot_eV"])

if frames >= total_frames:
    print(f"[{tag}] already finished", flush=True)
    raise SystemExit

atoms.calc = calc
dyn = make_dyn(atoms)  # RNG state restored from the last committed frame
traj = Trajectory(traj_path, "a")
logf = open(log_path, "a", newline="")
logw = csv.writer(logf)


def save():
    global frames
    a = atoms.copy()            # no calculator -> no forces in the file; momenta are kept for restart
    temp, energy = atoms.get_temperature(), atoms.get_potential_energy()
    if not np.isfinite([temp, energy]).all() or not np.isfinite(atoms.positions).all():
        raise RuntimeError("Non-finite state; last valid frame retained")
    a.info.update(rng_state=rng.bit_generator.state, T_inst_K=temp, Epot_eV=energy)
    traj.write(a)
    logw.writerow([frames, round(frames * args.every * args.dt / 1000, 3),
                   round(atoms.get_temperature(), 1), round(atoms.get_potential_energy(), 5)])
    logf.flush()
    frames += 1


if frames == 0:
    save()

info = {"tag": tag, "natoms": len(atoms), "T": args.T, "dt_fs": args.dt, "save_every": args.every,
        "equil_ps": args.equil_ps, "seed": args.seed, "total_frames": total_frames, "threads": nthreads}
t0, f0 = time.perf_counter(), frames
while frames < total_frames:
    dyn.run(args.every)
    save()
    if frames - f0 in (10, 100) or (frames - f0) % 1000 == 0:
        sps = (time.perf_counter() - t0) / ((frames - f0) * args.every)
        eta = sps * (total_frames - frames) * args.every / 3600
        info.update({"sec_per_step": round(sps, 4), "frames_done": frames,
                     "ETA_hours": round(eta, 1), "updated": time.strftime("%Y-%m-%d %H:%M")})
        json.dump(info, open(info_path, "w"), indent=1)
        print(f"[{tag}] frame {frames}/{total_frames}  {sps:.3f} s/step  ETA {eta:.1f} h  "
              f"T={atoms.get_temperature():.0f} K", flush=True)

traj.close()
logf.close()
info.update({"finished": True, "frames_done": frames})
json.dump(info, open(info_path, "w"), indent=1)
print(f"[{tag}] done", flush=True)
