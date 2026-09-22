import os
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write

DEFAULT_PATTERNS = ("*.xyz", "*.lammpstrj", "*.dump", "dump.*", "*.lammpsdump")


def _cell_and_origin(bounds):
    if len(bounds[0]) == 3:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo_b, zhi_b, yz) = bounds
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        zlo, zhi = zlo_b, zhi_b
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        xy = xz = yz = 0.0
    cell = np.array([[xhi - xlo, 0.0, 0.0],
                     [xy, yhi - ylo, 0.0],
                     [xz, yz, zhi - zlo]])
    return cell, np.array([xlo, ylo, zlo])


def is_dump(path):
    with open(path, errors="ignore") as f:
        return f.readline().startswith("ITEM: TIMESTEP")


def iter_dump(path, specorder=None):
    """Yield (positions, ids, symbols, cell, origin) per frame, sorted by atom id.

    Requires unwrapped xu/yu/zu columns.
    """
    natoms = cell = origin = None
    with open(path) as f:
        for line in f:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(f.readline())
            elif line.startswith("ITEM: BOX BOUNDS"):
                bounds = [[float(v) for v in f.readline().split()] for _ in range(3)]
                cell, origin = _cell_and_origin(bounds)
            elif line.startswith("ITEM: ATOMS"):
                cols = line.split()[2:]
                if not all(c in cols for c in ("xu", "yu", "zu")):
                    raise ValueError("no unwrapped xu/yu/zu columns")
                pos_idx = [cols.index(c) for c in ("xu", "yu", "zu")]
                id_idx = cols.index("id")
                rows = [f.readline().split() for _ in range(natoms)]

                ids = np.array([int(r[id_idx]) for r in rows])
                srt = np.argsort(ids)
                pos = np.array([[float(r[i]) for i in pos_idx] for r in rows])[srt]

                if "element" in cols:
                    j = cols.index("element")
                    symbols = [rows[k][j] for k in srt]
                elif "type" in cols and specorder:
                    j = cols.index("type")
                    symbols = [specorder[int(rows[k][j]) - 1] for k in srt]
                else:
                    raise ValueError(
                        "dump has no 'element' column; supply a species order "
                        "so 'type' can be mapped"
                    )
                yield pos, ids[srt], symbols, cell, origin


def process(path, specorder_user):
    n = 0
    mean = M2 = ids = symbols = cell = origin = None

    for pos, frame_ids, frame_symbols, frame_cell, frame_origin in \
            iter_dump(path, specorder_user):
        if n == 0:
            ids, symbols = frame_ids, frame_symbols
            cell, origin = frame_cell, frame_origin
            mean = np.zeros_like(pos)
            M2 = np.zeros_like(pos)
        elif pos.shape != mean.shape:
            raise ValueError("atom count changes between frames")
        n += 1
        delta = pos - mean
        mean += delta / n
        M2 += delta * (pos - mean)

    if n == 0:
        raise ValueError("no frames found")

    std_pos = np.sqrt(M2 / n)
    natoms = mean.shape[0]

    avg_atoms = Atoms(symbols=symbols, positions=mean - origin, cell=cell, pbc=True)
    avg_atoms.wrap()

    specorder = specorder_user or list(dict.fromkeys(symbols))
    parent = os.path.dirname(path) or "."

    data_out = os.path.join(parent, "avg_positions.data")
    std_out = os.path.join(parent, "std_dev.dat")
    for out in (data_out, std_out):
        if os.path.exists(out):
            print(f"  overwriting existing {out}")

    write(data_out, avg_atoms, format="lammps-data", specorder=specorder)

    with open(std_out, "w") as f:
        f.write(f"# source: {path}\n")
        f.write("# std dev of unwrapped coordinates\n")
        f.write("# atom_id symbol std_x std_y std_z\n")
        for atom_id, sym, s in zip(ids, symbols, std_pos):
            f.write(f"{atom_id} {sym} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}\n")

    return n, natoms, data_out, std_out


work_dir = input("Directory containing trajectories: ").strip()
specorder_in = input("Species order (comma-separated, blank = first-appearance order): ").strip()
patterns_in = input(f"Filename patterns (comma-separated, blank = {', '.join(DEFAULT_PATTERNS)}): ").strip()

specorder_user = [s.strip() for s in specorder_in.split(",")] if specorder_in else None
patterns = tuple(p.strip() for p in patterns_in.split(",")) if patterns_in else DEFAULT_PATTERNS

os.chdir(work_dir)
root = Path(".")

candidates = sorted({p for pat in patterns for p in root.rglob(pat) if p.is_file()})
files = [str(p) for p in candidates if is_dump(p)]
skipped = [str(p) for p in candidates if str(p) not in files]

if skipped:
    print(f"Not LAMMPS dump text, ignoring: {', '.join(skipped)}\n")
if not files:
    raise SystemExit(f"No LAMMPS dump files matching {patterns} under {os.getcwd()}")

print(f"Found {len(files)} dump file(s) under {os.getcwd()}\n")
n_ok = 0
for path in files:
    print(path)
    try:
        nframes, natoms, data_out, std_out = process(path, specorder_user)
    except Exception as e:
        print(f"  SKIPPED ({type(e).__name__}: {e})")
        continue
    n_ok += 1
    print(f"  {nframes} frames, {natoms} atoms -> {data_out}, {std_out}")

print(f"\nDone: {n_ok}/{len(files)} processed.")