import os
import numpy as np
from ase.io import read, write

traj_path = input("Path to MD trajectory file: ").strip()
out_dir = input("Output directory: ").strip()
specorder_in = input("Species order (comma-separated, blank = first-appearance order): ").strip()

os.makedirs(out_dir, exist_ok=True)

frames = read(traj_path, index=":")
nframes = len(frames)
natoms = len(frames[0])

positions = np.zeros((nframes, natoms, 3))
for i, atoms in enumerate(frames):
    atoms.wrap()
    positions[i] = atoms.get_positions()

avg_pos = positions.mean(axis=0)   # atoms x 3
std_pos = positions.std(axis=0)    # atoms x 3

ref = frames[0]
symbols = ref.get_chemical_symbols()

# preserve original atom id order (falls back to file order if no ids present)
if "id" in ref.arrays:
    ids = ref.get_array("id")
else:
    ids = np.arange(1, natoms + 1)
order = np.argsort(ids)

avg_atoms = ref.copy()[order]
avg_atoms.set_positions(avg_pos[order])
std_pos = std_pos[order]
symbols = [symbols[i] for i in order]
ids = ids[order]

# species order: user-supplied, else first-appearance order in original file
if specorder_in:
    specorder = [s.strip() for s in specorder_in.split(",")]
else:
    specorder = list(dict.fromkeys(symbols))

write(
    os.path.join(out_dir, "avg_positions.data"),
    avg_atoms,
    format="lammps-data",
    specorder=specorder,
)

std_path = os.path.join(out_dir, "std_dev.dat")
with open(std_path, "w") as f:
    f.write("# atom_id symbol std_x std_y std_z\n")
    for atom_id, sym, s in zip(ids, symbols, std_pos):
        f.write(f"{atom_id} {sym} {s[0]:.6f} {s[1]:.6f} {s[2]:.6f}\n")

print(f"Averaged over {nframes} frames, {natoms} atoms.")
print(f"Wrote {os.path.join(out_dir, 'avg_positions.data')}")
print(f"Wrote {std_path}")