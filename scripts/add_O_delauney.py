import numpy as np
import argparse
import matplotlib.pyplot as plt

from ase.io import read, write
from ase import Atoms
from ase.geometry import get_distances

from scipy.spatial import Delaunay

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--input", required=True)

parser.add_argument("--n_atoms", type=int, required=True)

parser.add_argument(
    "--site_types",
    nargs="+",
    choices=["fcc", "hcp", "bridge", "top"],
    default=["fcc", "hcp", "bridge", "top"]
)

parser.add_argument(
    "--height",
    type=float,
    default=1.8
)

parser.add_argument(
    "--d_min",
    type=float,
    default=2.0
)

parser.add_argument(
    "--z_tol",
    type=float,
    default=0.8
)

parser.add_argument(
    "--out",
    required=True,
    help="output lammps-data filename"
)

args = parser.parse_args()

# -----------------------------
# Load structure
# -----------------------------
atoms = read(args.input, format="lammps-data")
atoms.wrap()

slab = atoms.copy()
slab.wrap()

pos = slab.get_positions()
cell = slab.get_cell()
pbc = slab.get_pbc()

symbols = slab.get_chemical_symbols()

# -----------------------------
# Extract Ti atoms
# -----------------------------
ti_idx = [i for i, s in enumerate(symbols) if s == "Ti"]

ti_pos = pos[ti_idx]

# -----------------------------
# Detect Ti layers
# -----------------------------
ti_z = ti_pos[:, 2]
z_sorted = np.sort(ti_z)

layers = []
current = [z_sorted[0]]

for zi in z_sorted[1:]:

    if abs(zi - current[-1]) < args.z_tol:
        current.append(zi)

    else:
        layers.append(current)
        current = [zi]

layers.append(current)

layers = sorted(layers, key=lambda l: np.mean(l))

top_layer_z = np.mean(layers[-1])
second_layer_z = np.mean(layers[-2])

# -----------------------------
# Top-layer Ti atoms
# -----------------------------
top_ti_idx = [
    i for i in ti_idx
    if abs(pos[i, 2] - top_layer_z) < args.z_tol
]

top_ti_pos = pos[top_ti_idx]
top_xy = top_ti_pos[:, :2]

# -----------------------------
# Second-layer Ti atoms
# -----------------------------
second_ti_idx = [
    i for i in ti_idx
    if abs(pos[i, 2] - second_layer_z) < args.z_tol
]

second_ti_pos = pos[second_ti_idx]
second_xy = second_ti_pos[:, :2]

# -----------------------------
# PBC tiling
# -----------------------------
cell_xy = cell[:2, :2]

shifts = [-1, 0, 1]

def tile(xy):

    out = []

    for i in shifts:
        for j in shifts:

            shift = i * cell_xy[0] + j * cell_xy[1]

            out.append(xy + shift)

    return np.vstack(out)

all_xy = tile(top_xy)

# -----------------------------
# Delaunay triangulation
# -----------------------------
tri = Delaunay(all_xy)

# -----------------------------
# Unique helper
# -----------------------------
def unique_sites(sites, tol=1e-3):

    unique = []

    for s in sites:

        if not any(np.linalg.norm(s - u) < tol for u in unique):
            unique.append(s)

    return np.array(unique)

# -----------------------------
# Top sites
# -----------------------------
top_sites = top_xy.copy()

# -----------------------------
# Bridge sites
# -----------------------------
bridge_sites = []

edges = set()

for simplex in tri.simplices:

    for i in range(3):

        a = simplex[i]
        b = simplex[(i + 1) % 3]

        edge = tuple(sorted((a, b)))

        if edge in edges:
            continue

        edges.add(edge)

        mid = 0.5 * (all_xy[a] + all_xy[b])

        frac = np.linalg.solve(cell_xy.T, mid)

        if (0 <= frac[0] < 1) and (0 <= frac[1] < 1):
            bridge_sites.append(mid)

bridge_sites = unique_sites(np.array(bridge_sites))

# -----------------------------
# Hollow sites
# -----------------------------
hollow_sites = []

for simplex in tri.simplices:

    pts = all_xy[simplex]

    c = np.mean(pts, axis=0)

    frac = np.linalg.solve(cell_xy.T, c)

    if (0 <= frac[0] < 1) and (0 <= frac[1] < 1):
        hollow_sites.append(c)

hollow_sites = unique_sites(np.array(hollow_sites))

# -----------------------------
# Classify hollow sites
# -----------------------------
hcp_sites = []
fcc_sites = []

tol_xy = 0.6

for h in hollow_sites:

    found = False

    for s in second_xy:

        delta = h - s

        frac = np.linalg.solve(cell_xy.T, delta)

        frac -= np.round(frac)

        cart = frac @ cell_xy

        d = np.linalg.norm(cart)

        if d < tol_xy:

            hcp_sites.append(h)

            found = True
            break

    if not found:
        fcc_sites.append(h)

hcp_sites = np.array(hcp_sites)
fcc_sites = np.array(fcc_sites)

# -----------------------------
# Site map
# -----------------------------
site_map = {
    "top": top_sites,
    "bridge": bridge_sites,
    "hcp": hcp_sites,
    "fcc": fcc_sites
}

candidate_sites = np.vstack([
    site_map[s]
    for s in args.site_types
])

candidate_sites = unique_sites(candidate_sites)

# -----------------------------
# Distance validation
# -----------------------------
z_ads = top_layer_z + args.height

def is_valid(candidate_xy, chosen_xy):

    cand = np.array([
        [candidate_xy[0], candidate_xy[1], z_ads]
    ])

    ref = slab.get_positions().copy()

    if len(chosen_xy) > 0:

        ads = np.column_stack([
            chosen_xy,
            np.full(len(chosen_xy), z_ads)
        ])

        ref = np.vstack([ref, ads])

    _, dists = get_distances(
        cand,
        ref,
        cell=cell,
        pbc=True
    )

    dists = dists.flatten()

    dists = dists[dists > 1e-6]

    return np.all(dists >= args.d_min)

# -----------------------------
# Select sites
# -----------------------------
idx = np.arange(len(candidate_sites))
np.random.shuffle(idx)

chosen = []

for i in idx:

    s = candidate_sites[i]

    if is_valid(s, chosen):
        chosen.append(s)

    if len(chosen) == args.n_atoms:
        break

if len(chosen) < args.n_atoms:

    raise RuntimeError(
        "Could not place requested number of O atoms."
    )

chosen = np.array(chosen)

# -----------------------------
# Create adsorbates
# -----------------------------
new_pos = np.array([
    [x, y, z_ads]
    for x, y in chosen
])

new_atoms = Atoms(
    "O" * args.n_atoms,
    positions=new_pos,
    cell=cell,
    pbc=pbc
)

combined = atoms + new_atoms

# -----------------------------
# Enforce Ti then O ordering
# -----------------------------
ti = combined[[a.symbol == "Ti" for a in combined]]
o = combined[[a.symbol == "O" for a in combined]]

combined = ti + o

combined.set_cell(cell, scale_atoms=False)
combined.wrap()

# -----------------------------
# Write output structure
# -----------------------------
write(
    args.out,
    combined,
    format="lammps-data",
    atom_style="atomic",
    specorder=["Ti", "O"]
)

# -----------------------------
# DEBUG PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 9))

# top-layer Ti
ax.scatter(
    top_xy[:, 0],
    top_xy[:, 1],
    s=140,
    label="Top-layer Ti"
)

# second-layer Ti
ax.scatter(
    second_xy[:, 0],
    second_xy[:, 1],
    s=100,
    marker="s",
    alpha=0.5,
    label="Second-layer Ti"
)

# bridge
if len(bridge_sites) > 0:

    ax.scatter(
        bridge_sites[:, 0],
        bridge_sites[:, 1],
        marker="_",
        s=200,
        label="Bridge"
    )

# hcp
if len(hcp_sites) > 0:

    ax.scatter(
        hcp_sites[:, 0],
        hcp_sites[:, 1],
        marker="^",
        s=120,
        label="hcp"
    )

# fcc
if len(fcc_sites) > 0:

    ax.scatter(
        fcc_sites[:, 0],
        fcc_sites[:, 1],
        marker="x",
        s=120,
        label="fcc"
    )

# chosen adsorption sites
ax.scatter(
    chosen[:, 0],
    chosen[:, 1],
    s=250,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Chosen O sites"
)

# simulation cell
origin = np.array([0.0, 0.0])

a1 = cell_xy[0]
a2 = cell_xy[1]

corners = np.array([
    origin,
    a1,
    a1 + a2,
    a2,
    origin
])

ax.plot(corners[:, 0], corners[:, 1])

# triangulation
ax.triplot(
    all_xy[:, 0],
    all_xy[:, 1],
    tri.simplices,
    alpha=0.25
)

# ax.set_aspect("equal")

# ax.set_xlabel("x (Å)")
# ax.set_ylabel("y (Å)")
xmin = np.min(corners[:, 0])
xmax = np.max(corners[:, 0])

ymin = np.min(corners[:, 1])
ymax = np.max(corners[:, 1])

pad = 1.0

ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

ax.set_aspect("equal")

ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")

ax.legend()

plt.tight_layout()

png_out = args.out.replace(".data", ".png")

plt.savefig(
    png_out,
    dpi=300
)

# plt.show()

print(f"Structure written to: {args.out}")
print(f"Debug plot written to: {png_out}")