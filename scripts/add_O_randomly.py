import numpy as np
import argparse

from ase.io import read, write
from ase import Atoms
from ase.geometry import get_distances

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)

parser.add_argument(
    "--n_atoms",
    type=int,
    required=True,
    help="Number of O atoms to place"
)

parser.add_argument(
    "--height",
    type=float,
    default=1.8,
    help="Height above average top Ti plane"
)

parser.add_argument(
    "--d_min",
    type=float,
    default=2.0,
    help="Minimum allowed distance (Å) between O and any atom, including O-O"
)

parser.add_argument(
    "--z_tol",
    type=float,
    default=0.8,
    help="Tolerance for identifying Ti layers"
)

parser.add_argument(
    "--max_attempts",
    type=int,
    default=50000,
    help="Maximum random placement attempts"
)

parser.add_argument(
    "--out",
    required=True,
    help="Output LAMMPS data file"
)

args = parser.parse_args()

# -----------------------------
# Load structure
# -----------------------------
atoms = read(args.input, format="lammps-data")

atoms.wrap()

cell = atoms.get_cell()
pbc = atoms.get_pbc()

pos = atoms.get_positions()
symbols = atoms.get_chemical_symbols()

# -----------------------------
# Extract Ti atoms
# -----------------------------
ti_idx = [i for i, s in enumerate(symbols) if s == "Ti"]

if len(ti_idx) == 0:
    raise RuntimeError("No Ti atoms found.")

ti_pos = pos[ti_idx]

# -----------------------------
# Detect Ti layers
# -----------------------------
ti_z = np.sort(ti_pos[:, 2])

layers = []
current = [ti_z[0]]

for z in ti_z[1:]:

    if abs(z - current[-1]) < args.z_tol:
        current.append(z)

    else:
        layers.append(current)
        current = [z]

layers.append(current)

layers = sorted(layers, key=lambda x: np.mean(x))

top_layer_z = np.mean(layers[-1])

print(f"Average top Ti layer z = {top_layer_z:.4f} Å")

# -----------------------------
# Oxygen adsorption height
# -----------------------------
z_ads = top_layer_z + args.height

# -----------------------------
# Random sampling helpers
# -----------------------------
cell_xy = cell[:2, :2]

def random_xy_in_cell():
    """
    Generate random xy position inside periodic cell.
    """

    frac = np.random.rand(2)

    xy = frac @ cell_xy

    return xy

# -----------------------------
# MIC distance check
# -----------------------------
def is_valid(candidate_xy, chosen_xy):
    """
    Check whether candidate O position satisfies
    minimum distance criterion under MIC.
    """

    candidate = np.array([
        [candidate_xy[0], candidate_xy[1], z_ads]
    ])

    ref = atoms.get_positions().copy()

    # add already placed O atoms
    if len(chosen_xy) > 0:

        ads = np.column_stack([
            chosen_xy,
            np.full(len(chosen_xy), z_ads)
        ])

        ref = np.vstack([ref, ads])

    _, dists = get_distances(
        candidate,
        ref,
        cell=cell,
        pbc=True
    )

    dists = dists.flatten()

    # remove self-like numerical zeroes
    dists = dists[dists > 1e-8]

    return np.all(dists >= args.d_min)

# -----------------------------
# Random O placement
# -----------------------------
chosen = []

attempts = 0

while len(chosen) < args.n_atoms:

    attempts += 1

    if attempts > args.max_attempts:

        raise RuntimeError(
            f"Could not place {args.n_atoms} O atoms "
            f"after {args.max_attempts} attempts."
        )

    trial_xy = random_xy_in_cell()

    if is_valid(trial_xy, chosen):

        chosen.append(trial_xy)

        print(
            f"Placed O #{len(chosen)} "
            f"after {attempts} attempts"
        )

chosen = np.array(chosen)

# -----------------------------
# Create O atoms
# -----------------------------
o_positions = np.column_stack([
    chosen,
    np.full(len(chosen), z_ads)
])

o_atoms = Atoms(
    symbols="O" * args.n_atoms,
    positions=o_positions,
    cell=cell,
    pbc=pbc
)

# -----------------------------
# Combine structure
# -----------------------------
combined = atoms + o_atoms

# enforce Ti then O ordering
ti_atoms = combined[[a.symbol == "Ti" for a in combined]]
o_atoms = combined[[a.symbol == "O" for a in combined]]

combined = ti_atoms + o_atoms

combined.set_cell(cell, scale_atoms=False)

combined.wrap()

# -----------------------------
# Write output
# -----------------------------
write(
    args.out,
    combined,
    format="lammps-data",
    atom_style="atomic",
    specorder=["Ti", "O"]
)

print(f"\nStructure written to: {args.out}")