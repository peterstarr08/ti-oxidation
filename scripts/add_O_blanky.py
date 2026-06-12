import argparse
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from ase import Atoms
from scipy.spatial import cKDTree

from scipy.spatial import Delaunay
from sklearn.cluster import HDBSCAN
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="Construct exposed surface mesh."
)

parser.add_argument(
    "path",
    help="LAMMPS data file"
)
parser.add_argument(
    "-n",
    type=int,
    required=True,
    help="Number of O atoms to insert"
)

parser.add_argument(
    "--height",
    type=float,
    default=1.5
)

parser.add_argument(
    "--min-distance",
    type=float,
    default=2.2
)
args = parser.parse_args()

path = args.path


# ============================================================
# Read structure
# ============================================================

atoms = read(
    path,
    ":",
    format="lammps-data"
)

snapshot = atoms[0]
snapshot.wrap()

cell = snapshot.cell.lengths()

lx = cell[0]
ly = cell[1]


# ============================================================
# Find Ti layers
# ============================================================

ti_atoms = Atoms([
    atom
    for atom in snapshot
    if atom.symbol == "Ti"
])

ti_pos = ti_atoms.get_positions()

ti_z = ti_pos[:, 2].reshape(-1, 1)

clustering = HDBSCAN(
    min_cluster_size=50,
    max_cluster_size=192
).fit(ti_z)

labels = clustering.labels_

valid_labels = [
    l
    for l in np.unique(labels)
    if l != -1
]

layer_info = []

for label in valid_labels:

    xyz = ti_pos[labels == label]

    layer_info.append(
        (
            label,
            xyz[:, 2].mean()
        )
    )

layer_info.sort(
    key=lambda x: x[1],
    reverse=True
)

top_cluster_label = layer_info[0][0]

top_layer_xyz = ti_pos[
    labels == top_cluster_label
]

mean_top_z = top_layer_xyz[:, 2].mean()

print(
    f"Top layer cluster = {top_cluster_label}"
)

print(
    f"Mean top layer z = {mean_top_z:.3f}"
)


# ============================================================
# Exposed atoms
# ============================================================

# surface_points = []

# # Top layer Ti

# for atom in ti_atoms:

#     idx = atom.index

#     if labels[idx] == top_cluster_label:

#         surface_points.append(
#             atom.position.copy()
#         )

# # Noise Ti above top layer

# noise_mask = labels == -1

# for atom, label in zip(
#     ti_atoms,
#     labels
# ):

#     if (
#         label == -1
#         and atom.position[2] > mean_top_z
#     ):

#         surface_points.append(
#             atom.position.copy()
#         )

# # O atoms above top layer

# for atom in snapshot:

#     if atom.symbol != "O":
#         continue

#     if atom.position[2] > mean_top_z:

#         surface_points.append(
#             atom.position.copy()
#         )

# surface_points = np.array(
#     surface_points
# )

# print(
#     f"Surface atoms used: {len(surface_points)}"
# )

surface_points = []

# top layer Ti

for atom, label in zip(ti_atoms, labels):

    if label == top_cluster_label:

        surface_points.append(
            atom.position.copy()
        )

top_layer_min_z = np.min(
    top_layer_xyz[:,2]
)

# noise Ti only

for atom, label in zip(
    ti_atoms,
    labels
):

    if (
        label == -1
        and atom.position[2] > top_layer_min_z
    ):

        surface_points.append(
            atom.position.copy()
        )

surface_points = np.array(
    surface_points
)

print(
    f"Surface points = {len(surface_points)}"
)


# ============================================================
# Plot exposed atoms
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(
    surface_points[:, 0],
    surface_points[:, 1],
    s=8
)

ax.set_aspect("equal")

ax.set_title(
    "Surface atoms"
)

plt.tight_layout()

plt.savefig(
    "surface_atoms.png",
    dpi=300
)

plt.close()


# ============================================================
# Build periodic mesh
# ============================================================

shifts = [
    (-1, -1),
    (-1,  0),
    (-1,  1),
    ( 0, -1),
    ( 0,  0),
    ( 0,  1),
    ( 1, -1),
    ( 1,  0),
    ( 1,  1)
]

replicated = []

for sx, sy in shifts:

    pts = surface_points.copy()

    pts[:, 0] += sx * lx
    pts[:, 1] += sy * ly

    replicated.append(pts)

replicated = np.vstack(
    replicated
)

tri = Delaunay(
    replicated[:, :2]
)

print(
    f"Triangles = {len(tri.simplices)}"
)


# ============================================================
# Top-view mesh
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.triplot(
    replicated[:, 0],
    replicated[:, 1],
    tri.simplices,
    linewidth=0.3
)

ax.scatter(
    surface_points[:, 0],
    surface_points[:, 1],
    s=5
)

ax.set_aspect(
    "equal"
)

ax.set_title(
    "Periodic blanket mesh"
)

plt.tight_layout()

plt.savefig(
    "surface_mesh_topview.png",
    dpi=300
)

plt.close()


# ============================================================
# Central-cell triangles only
# ============================================================

central_faces = []

for simplex in tri.simplices:

    verts = replicated[
        simplex
    ]

    center = verts.mean(
        axis=0
    )

    if (
        0 <= center[0] < lx
        and
        0 <= center[1] < ly
    ):

        central_faces.append(
            verts
        )

print(
    f"Central triangles = {len(central_faces)}"
)

def triangle_area(v0, v1, v2):

    return 0.5 * np.linalg.norm(
        np.cross(
            v1 - v0,
            v2 - v0
        )
    )
    

areas = np.array([
    triangle_area(
        face[0],
        face[1],
        face[2]
    )
    for face in central_faces
])

area_prob = areas / areas.sum()


# ============================================================
# 3D mesh plot
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

mesh = Poly3DCollection(
    central_faces,
    alpha=0.7,
    linewidths=0.15
)

ax.add_collection3d(
    mesh
)

ax.scatter(
    surface_points[:, 0],
    surface_points[:, 1],
    surface_points[:, 2],
    s=3
)

ax.set_xlim(
    0,
    lx
)

ax.set_ylim(
    0,
    ly
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(
    "Surface blanket mesh"
)

plt.tight_layout()

plt.savefig(
    "surface_mesh_3d.png",
    dpi=300
)

plt.close()


print()
print("Saved:")
print("  surface_atoms.png")
print("  surface_mesh_topview.png")
print("  surface_mesh_3d.png")


def sample_triangle(face):

    r1 = np.random.rand()
    r2 = np.random.rand()

    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2

    point = (
        u * face[0]
        + v * face[1]
        + w * face[2]
    )

    return point


def triangle_normal(face):

    v0, v1, v2 = face

    n = np.cross(
        v1 - v0,
        v2 - v0
    )

    n /= np.linalg.norm(n)

    if n[2] < 0:
        n *= -1

    return n


def valid_position(
    candidate,
    existing_positions,   # numpy array, pre-extracted once
    new_positions,
    min_distance,
    cell
):
    lx, ly, lz = cell

    all_positions = (
        np.vstack([existing_positions, new_positions])
        if len(new_positions) > 0
        else existing_positions
    )

    # Vectorised MIC distance — no ASE, no copy
    delta = all_positions - candidate
    delta[:, 0] -= np.round(delta[:, 0] / lx) * lx
    delta[:, 1] -= np.round(delta[:, 1] / ly) * ly
    delta[:, 2] -= np.round(delta[:, 2] / lz) * lz

    dists = np.linalg.norm(delta, axis=1)

    return dists.min() > min_distance


def insert_oxygen(
    snapshot,
    central_faces,
    area_prob,
    n_oxygen,
    height,
    min_distance,
    max_attempts=200000       # bump this up
):
    existing_positions = snapshot.get_positions()
    cell = snapshot.cell.lengths()

    new_positions = []
    attempts = 0

    while (
        len(new_positions) < n_oxygen
        and attempts < max_attempts
    ):
        attempts += 1

        face_index = np.random.choice(
            len(central_faces),
            p=area_prob
        )

        face = central_faces[face_index]
        point = sample_triangle(face)
        normal = triangle_normal(face)
        candidate = point + height * normal

        if valid_position(
            candidate,
            existing_positions,
            new_positions,
            min_distance,
            cell
        ):
            new_positions.append(candidate)
            print(f"  Placed {len(new_positions)}/{n_oxygen}", end="\r")

    print()

    if len(new_positions) < n_oxygen:
        print(
            f"WARNING: only placed {len(new_positions)}/{n_oxygen} atoms "
            f"after {max_attempts} attempts.\n"
            f"Try reducing --min-distance (currently {min_distance}) "
            f"or --n (currently {n_oxygen})."
        )
    else:
        print(f"Placed {len(new_positions)} of {n_oxygen} (attempts: {attempts})")

    return np.array(new_positions) if new_positions else np.empty((0, 3))

new_oxygen = insert_oxygen(
    snapshot=snapshot,
    central_faces=central_faces,
    area_prob=area_prob,
    n_oxygen=args.n,
    height=args.height,
    min_distance=args.min_distance
)

if len(new_oxygen) == 0:
    print("No oxygen atoms placed. Exiting.")
    raise SystemExit(1)

from ase import Atom

# Replace the loop that was using Atoms(symbols="O", positions=pos)
new_snapshot = snapshot.copy()

for pos in new_oxygen:
    new_snapshot.append(Atom('O', position=pos))

# Remove the stale 'type' array inherited from the LAMMPS read.
# If left in place, ASE's writer uses it directly and new atoms
# get type=0 since they were never assigned a LAMMPS type.
if 'type' in new_snapshot.arrays:
    del new_snapshot.arrays['type']


all_new_indices = range(
    len(snapshot),
    len(new_snapshot)
)

global_min = 1e9

for idx in all_new_indices:

    distances = new_snapshot.get_distances(
        idx,
        range(idx),
        mic=True
    )

    global_min = min(
        global_min,
        np.min(distances)
    )

print()
print(
    f"Requested minimum distance : "
    f"{args.min_distance:.3f}"
)

print(
    f"Actual minimum distance    : "
    f"{global_min:.3f}"
)

if global_min >= args.min_distance:
    print("PASS")
else:
    print("FAIL")
    
    
    
write(
    "inserted_structure.data",
    new_snapshot,
    format="lammps-data",
    specorder=["Ti", "O"]
)



fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(
    111,
    projection="3d"
)

mesh = Poly3DCollection(
    central_faces,
    alpha=0.4,
    linewidths=0.1
)

ax.add_collection3d(mesh)

ax.scatter(
    new_oxygen[:,0],
    new_oxygen[:,1],
    new_oxygen[:,2],
    s=20,
    label="Inserted O"
)

ax.set_xlim(0, lx)
ax.set_ylim(0, ly)

ax.legend()

plt.tight_layout()

plt.savefig(
    "oxygen_inserted.png",
    dpi=300
)