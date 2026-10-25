"""А тут попытка сравнить миграцию одного иона лития с миграцией сразу нескольких(concerted) вдоль одного канала LGPS.
Что в инпут
 1. Нужна структура в формате .cif
 2. Создает суперячейку размером не менее 20 А вдоль оси c и находит столбец Li с наибольшей плотностью вдоль оси c (c-канал).
 3. Удаляет один канал Li (вакансию). Для n = 1..NMAX каждый из n соседних каналов Li смещается на одну
    занятую ячейку в сторону вакансии:  n = 1 -> одношаговая миграция, n >= 2 -> согласованная миграция.
 4. Оптимизирует концы траектории.
 5.Экспортные данные для каждого n:
      sevennet.csv, sevennet_path.traj        — энергетический профиль и изображения Sevennet
      vasp_neb/00..NN/POSCAR, INCAR, KPOINTS  — входные данные для DFT-NEB, траектория Sevennet в качестве начального приближения
      vasp_sp/img_XX/POSCAR, INCAR, KPOINTS   - отдельные точки DFT на каждом изображении Sevennet
    и файл summary.json со всеми барьерами.

Использование:  python neb_lgps.py --nmax 4 --nimages 7
POTCAR-файлы не записываются

"""
import argparse, os, json, csv, math
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
try:
    from ase.mep import NEB
except ImportError:  # ase < 3.23
    from ase.neb import NEB

p = argparse.ArgumentParser()
p.add_argument("--nmax", type=int, default=4, help="largest number of ions moving together")
p.add_argument("--nimages", type=int, default=7, help="intermediate NEB images")
p.add_argument("--start", type=int, default=0, help="which channel site becomes the vacancy")
p.add_argument("--out", default="neb_out")
p.add_argument("--calc", default="mace", choices=["mace", "sevennet", "lj"], help="lj = pipeline test only")
p.add_argument("--structure", default="structures/LGPS.cif")
args = p.parse_args()

SPECIES = ["Li", "Ge", "P", "S"]          # POSCAR / POTCAR order

# ---------------- calculator ----------------
def make_calc():
    if args.calc == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=2.0, epsilon=0.01, rc=5.0)
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    if args.calc == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator("7net-0", device="cpu")
    from mace.calculators import mace_mp
    return mace_mp(model="medium", device="cpu", default_dtype="float64")

calc = make_calc()

# ---------------- structure ----------------
if not os.path.exists(args.structure):
    raise SystemExit("Не найдена исходная структура: " + args.structure)

atoms = read(args.structure)
if len(atoms) != 50:
    raise SystemExit("Ожидалась исходная ячейка LGPS из 50 атомов")
axis = 2
rep = [2, 1, 2]
atoms = atoms.repeat(rep)
# group atoms by species so that POSCARs are valid (Li, Ge, P, S)
syms = atoms.get_chemical_symbols()
order = sorted(range(len(atoms)), key=lambda i: (SPECIES.index(syms[i]) if syms[i] in SPECIES else 99, i))
atoms = atoms[order]
print(f"supercell {rep}, {len(atoms)} atoms, channel axis = {'abc'[axis]}, lengths = {np.round(atoms.cell.lengths(), 2)}")

# ---------------- find the c-channel ----------------
'''Идея: атомы Li, которые лежат друг над другом вдоль c, имеют почти одинаковые координаты в плоскости ab. Значит, если сгруппировать Li по их проекции на ab, получатся колонки, то есть каналы.'''
def li_columns(at, axis, tol=1.5):
    frac = at.get_scaled_positions() # дробные координаты 
    ip = [i for i in range(3) if i != axis]  # для axis=2 [0, 1], то есть a и b
    cell_ip = np.array(at.cell)[ip]  #  # векторы a и b, матрица 2x3, перпендикулярны каналу
    cols = []
    for i in (a.index for a in at if a.symbol == "Li"):
        for col in cols:
            d = frac[i][ip] - frac[col[0]][ip]
            d -= np.round(d) #  периодичность
            if np.linalg.norm(d @ cell_ip) < tol: # перевод в вектор
                col.append(i) # если расстояние меньше порога, атом принадлежит колонке
                break
        else:
            cols.append([i]) # цикл перебирает найденные колонки и сравнивает проекцию на ab с первой. 
    cols.sort(key=len, reverse=True)
    return [sorted(c, key=lambda i: frac[i][axis] % 1.0) for c in cols]
'''  да это дурацкий алгоритм и он сработал только на небольшом тетрагональном LGPS. Зато работает'''
cols = li_columns(atoms, axis)
chan = cols[0] # берем самую заселенную колонку
chan = chan[args.start:] + chan[:args.start] # двигаем цепочку
zs = [round(float(atoms.get_scaled_positions()[i][axis] % 1 * atoms.cell.lengths()[axis]), 2) for i in chan] # Направление движения: Li с chan[k] переходит на место chan[k-1], то есть ионы сдвигаются в сторону меньших z, к вакансии.
print(f"Li columns along the axis (sizes): {[len(c) for c in cols]}")
print(f"chosen channel: {len(chan)} Li at z = {zs} A")
if args.nmax + 1 > len(chan):
    raise SystemExit(f"channel has only {len(chan)} Li; use --nmax <= {len(chan) - 1}")
# Декартовы z позиций канала и размеры всех колонок в лог. Так можно проверить, что канал найден правильно и позиции идут с разумным шагом.
# ---------------- helpers ----------------
# оптимизация концов
def relax(at, fmax=0.02):
    at.calc = calc
    ok = BFGS(at, logfile="-").run(fmax=fmax, steps=400)
    if not ok:
        raise RuntimeError("Релаксация конца NEB не сошлась")
    return at

def path_coordinate(images):
    s = [0.0]
    for a, b in zip(images[:-1], images[1:]):
        d, _ = find_mic(b.positions - a.positions, a.cell, pbc=True)
        s.append(s[-1] + float(np.linalg.norm(d)))
    return s

def kpoints_text(at):
    k = [max(1, round(20.0 / l)) for l in at.cell.lengths()]
    return f"Gamma-centred\n0\nGamma\n{k[0]} {k[1]} {k[2]}\n0 0 0\n"

INCAR_COMMON = """PREC = Accurate
ENCUT = 520
EDIFF = 1E-5
ISMEAR = 0
SIGMA = 0.05
ISPIN = 1            ! non-spin-polarised PBE, as in Dembitskiy et al. 2025
LREAL = Auto
ALGO = Fast
NELM = 120
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
INCAR_NEB = INCAR_COMMON + """IMAGES = {nimg}
SPRING = -5
LCLIMB = .TRUE.      ! VTST; without VTST remove LCLIMB/IOPT and use IBRION = 1
IBRION = 3
POTIM = 0
IOPT = 1
ISIF = 2
NSW = 300
EDIFFG = -0.05
"""
INCAR_SP = INCAR_COMMON + "IBRION = -1\nNSW = 0\n"

def wposcar(path, at):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write(path, at, format="vasp", direct=True, sort=False)

# ---------------- main loop ----------------
'''это основная программа'''
os.makedirs(args.out, exist_ok=True)
summary = {"supercell": rep, "natoms_with_vacancy": len(atoms) - 1, "channel_z_A": zs, "runs": {}}

for n in range(1, args.nmax + 1):
    tag = f"n{n}"
    d = os.path.join(args.out, tag)
    print(f"\n=== {tag}: {n} Li moving ===")

    ini = atoms.copy() # построение начала  конца 
    fin = atoms.copy()
    for k in range(1, n + 1):                       # Li at site k -> site k-1
        fin.positions[chan[k]] = atoms.positions[chan[k - 1]]
    vac = chan[0]
    del ini[vac] # Удаляется Li с позиции 0 
    del fin[vac]

    relax(ini) # релаксация 
    relax(fin)
    moved = [c - (1 if c > vac else 0) for c in chan[1:n + 1]] '''После del все атомы с индексом больше vac сдвинулись на единицу вниз. Эта строка пересчитывает индексы движущихся Li под новую нумерацию.'''
    shift = [float(np.linalg.norm(find_mic(fin.positions[m] - ini.positions[m], fin.cell)[0])) for m in moved]
    '''Расстояние, на которое каждый движущийся Li отстоит в fin от своего положения в ini после релаксации. Ожидается примерно расстояние между соседними позициями в канале. Если какое-то значение около нуля, ион при релаксации скатился обратно, конечное состояние совпало с начальным, и NEB для этого n вообще не имеет смысла.'''
    print("displacement of moving Li after end-point relaxation (A):", np.round(shift, 2))

    images = [ini] + [ini.copy() for _ in range(args.nimages)] + [fin]
    for im in images:
        im.calc = calc
    neb = NEB(images, climb=False, method="improvedtangent", allow_shared_calculator=True)
    neb.interpolate(method="idpp", mic=True)
    opt = FIRE(neb, logfile=os.path.join(args.out, f"{tag}_neb.log"))
    opt.run(fmax=0.10, steps=400)
    neb.climb = True
    converged = opt.run(fmax=0.05, steps=600)

    E = np.array([im.get_potential_energy() for im in images])
    E -= E[0]
    s = path_coordinate(images)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "sevennet_path.traj"), images)
    with open(os.path.join(d, "sevennet_profile.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "path_A", "E_minus_E0_eV"])
        for i, (si, ei) in enumerate(zip(s, E)):
            w.writerow([i, round(si, 4), round(float(ei), 5)])  # это профили энергии сохранились

    # VASP inputs: NEB with MACE path as initial guess, and single points on MACE images
    nimg = args.nimages
    for i, im in enumerate(images):
        wposcar(os.path.join(d, "vasp_neb", f"{i:02d}", "POSCAR"), im)
        sp = os.path.join(d, "vasp_sp", f"img_{i:02d}")
        wposcar(os.path.join(sp, "POSCAR"), im)
        open(os.path.join(sp, "INCAR"), "w").write(INCAR_SP)
        open(os.path.join(sp, "KPOINTS"), "w").write(kpoints_text(im))
    open(os.path.join(d, "vasp_neb", "INCAR"), "w").write(INCAR_NEB.format(nimg=nimg))
    open(os.path.join(d, "vasp_neb", "KPOINTS"), "w").write(kpoints_text(ini))

    res = {"n_moving": n, "barrier_forward_eV": round(float(E.max()), 4),
           "barrier_backward_eV": round(float(E.max() - E[-1]), 4),
           "dE_final_eV": round(float(E[-1]), 4), "converged": bool(converged),
           "endpoint_shift_A": [round(x, 2) for x in shift]}
    summary["runs"][tag] = res
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(res))

with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nbarriers (eV):", {k: v["barrier_forward_eV"] for k, v in summary["runs"].items()})
''' как читать выдачу: 
Главное - прямой барьер при n = 1 по сравнению n ≥ 2. Если согласованный барьер ниже одиночного, согласованный механизм энергетически выгоднее.
Обязательно смотреть на форму профиля, а не только на максимум. Задан согласованный путь только начальным приближением, а NEB ищет путь минимальной энергии. Если на профиле при n ≥ 2 появились промежуточные минимумы, путь распался на последовательные прыжки, и это уже не согласованы. При семи изображениях для n = 4 такую структуру можно пропустить(.
converged = false или shift около нуля означают, что барьер для этого n интерпретировать нельзя.
