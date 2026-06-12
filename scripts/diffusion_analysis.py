from sklearn.cluster import HDBSCAN
import re
from ase.io import read
import pandas as pd
from scipy.spatial import Delaunay
from matplotlib.lines import Line2D
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="Analyze O atom positions between Ti layers across a LAMMPS MD trajectory."
)

parser.add_argument(
    "path",
    help="Path to LAMMPS dump trajectory file (.dump / .lammpstrj)"
)

parser.add_argument(
    "--every",
    type=int,
    default=40,
    metavar="N",
    help="Analyze every Nth snapshot by index (default: 1, i.e. every snapshot)"
)

parser.add_argument(
    "--min-cluster-size",
    type=int,
    default=50,
    dest="min_cluster_size",
    help="HDBSCAN min_cluster_size for Ti layer detection (default: 50)"
)

parser.add_argument(
    "--max-cluster-size",
    type=int,
    default=192,
    dest="max_cluster_size",
    help="HDBSCAN max_cluster_size for Ti layer detection (default: 192)"
)

parser.add_argument(
    "--atoms-per-ml",
    type=int,
    default=48,
    dest="atoms_per_ml",
    metavar="N",
    help="Number of O atoms per ML block for ML-resolved statistics (default: 48)"
)

parser.add_argument(
    "--outdir",
    type=str,
    default="traj_analysis",
    help="Root output directory (default: traj_analysis)"
)

args = parser.parse_args()

# ============================================================
# Output directory layout
# ============================================================
#
#  <outdir>/
#    images/
#      ti_layers/        <- ti_layers_step_XXXXXXXX.png
#      oxygen_regions/   <- oxygen_regions_step_XXXXXXXX.png
#    oxygen_statistics.csv              (all steps appended)
#    oxygen_statistics_by_ml.csv        (all steps appended)

OUTDIR          = args.outdir
IMG_DIR         = os.path.join(OUTDIR, "images")
TI_IMG_DIR      = os.path.join(IMG_DIR, "ti_layers")
OXY_IMG_DIR     = os.path.join(IMG_DIR, "oxygen_regions")
STATS_CSV       = os.path.join(OUTDIR, "oxygen_statistics.csv")
ML_STATS_CSV    = os.path.join(OUTDIR, "oxygen_statistics_by_ml.csv")

for d in [TI_IMG_DIR, OXY_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Read full trajectory
# ============================================================

print(f"Reading trajectory: {args.path}", flush=True)

all_snapshots = read(
    args.path,
    index=":",
    format="lammps-dump-text"
)

total = len(all_snapshots)
print(f"Total snapshots found: {total}", flush=True)

# Select every Nth snapshot by index
selected_indices = list(range(0, total, args.every))
print(f"Snapshots to analyze:  {len(selected_indices)}  (every {args.every})", flush=True)


# ============================================================
# Helpers: cluster plotter
# ============================================================

def plot_clusters(snapshot: Atoms, labels, filepath: str):
    ti_atoms = Atoms([a for a in snapshot if a.symbol == "Ti"])
    pos = ti_atoms.get_positions()

    unique_labels = np.unique(labels)
    cmap = plt.cm.tab20

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    legend_elements = []

    for i, label in enumerate([l for l in unique_labels if l != -1]):
        mask = labels == label
        color = cmap(i % 20)
        ax1.scatter(pos[mask, 0], pos[mask, 2], s=5, color=color)
        ax2.scatter(pos[mask, 1], pos[mask, 2], s=5, color=color)
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color=color,
                   label=f'Cluster {label} ({mask.sum()})')
        )

    noise_mask = labels == -1
    if np.any(noise_mask):
        ax1.scatter(pos[noise_mask, 0], pos[noise_mask, 2], s=5, color='black')
        ax2.scatter(pos[noise_mask, 1], pos[noise_mask, 2], s=5, color='black')
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color='black',
                   label=f'Noise ({noise_mask.sum()})')
        )

    ax1.set_xlabel("x (Å)");  ax1.set_ylabel("z (Å)");  ax1.set_title("x–z projection")
    ax2.set_xlabel("y (Å)");  ax2.set_ylabel("z (Å)");  ax2.set_title("y–z projection")
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.18, 0.5), loc='center right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Helpers: PBC replicated Delaunay surface
# ============================================================

def build_periodic_surface(xyz, lx, ly):
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1), ( 0, 0), ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]
    replicated_xy  = []
    replicated_xyz = []
    for sx, sy in shifts:
        shifted = xyz.copy()
        shifted[:, 0] += sx * lx
        shifted[:, 1] += sy * ly
        replicated_xy.append(shifted[:, :2])
        replicated_xyz.append(shifted)
    replicated_xy  = np.vstack(replicated_xy)
    replicated_xyz = np.vstack(replicated_xyz)
    tri = Delaunay(replicated_xy)
    return {"tri": tri, "xyz": replicated_xyz}


# ============================================================
# Helpers: barycentric interpolation
# ============================================================

def surface_height(surface, x, y):
    tri = surface["tri"]
    xyz = surface["xyz"]
    simplex = tri.find_simplex([[x, y]])[0]
    if simplex == -1:
        return np.nan
    transform = tri.transform[simplex]
    bary = np.dot(transform[:2], np.array([x, y]) - transform[2])
    bary = np.append(bary, 1 - bary.sum())
    verts = xyz[tri.simplices[simplex]]
    return np.sum(bary * verts[:, 2])


# ============================================================
# Helpers: build ordered layers
# ============================================================

def build_layers(snapshot, labels):
    ti_atoms = Atoms([a for a in snapshot if a.symbol == "Ti"])
    pos  = ti_atoms.get_positions()
    cell = snapshot.cell.lengths()
    lx, ly = cell[0], cell[1]

    temp = []
    for label in sorted([l for l in np.unique(labels) if l != -1]):
        xyz    = pos[labels == label]
        mean_z = xyz[:, 2].mean()
        temp.append((mean_z, xyz))

    temp.sort(key=lambda x: x[0], reverse=True)

    layers = {}
    for idx, (_, xyz) in enumerate(temp):
        layers[idx + 1] = build_periodic_surface(xyz, lx, ly)
    return layers


# ============================================================
# Helpers: classify oxygens
# ============================================================

def classify_oxygen_atoms(snapshot, layers):
    oxygen_atoms = Atoms([a for a in snapshot if a.symbol == "O"])
    oxygen_pos   = oxygen_atoms.get_positions()
    n_layers     = len(layers)

    counts = {"Above L1": 0}
    for i in range(1, n_layers):
        counts[f"L{i}-L{i+1}"] = 0
    counts[f"Below L{n_layers}"] = 0

    assignments = []

    for idx, pos in enumerate(oxygen_pos):
        x, y, z = pos
        heights = [surface_height(layers[l], x, y) for l in range(1, n_layers + 1)]
        global_id = oxygen_atoms[idx].index + 1

        if z > heights[0]:
            region = "Above L1"
            counts[region] += 1
            assignments.append((idx, global_id, region))
            continue

        assigned = False
        for i in range(n_layers - 1):
            if heights[i] > z > heights[i + 1]:
                region = f"L{i+1}-L{i+2}"
                counts[region] += 1
                assignments.append((idx, global_id, region))
                assigned = True
                break

        if not assigned:
            region = f"Below L{n_layers}"
            counts[region] += 1
            assignments.append((idx, global_id, region))

    return assignments, counts


# ============================================================
# Helpers: plot oxygen assignments
# ============================================================

def plot_oxygen_regions(snapshot, assignments, filepath: str):
    oxy = Atoms([a for a in snapshot if a.symbol == "O"])
    pos = oxy.get_positions()

    regions = sorted(set(r for _, _, r in assignments))
    cmap    = plt.cm.tab20
    region_colors = {region: cmap(i % 20) for i, region in enumerate(regions)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    legend_elements = []

    for region in regions:
        idxs = [idx for idx, _, r in assignments if r == region]
        xyz  = pos[idxs]
        color = region_colors[region]
        ax1.scatter(xyz[:, 0], xyz[:, 2], color=color, s=10)
        ax2.scatter(xyz[:, 1], xyz[:, 2], color=color, s=10)
        legend_elements.append(
            Line2D([0], [0], marker='o', linestyle='', color=color,
                   label=f"{region} ({len(idxs)})")
        )

    ax1.set_xlabel("x (Å)");  ax1.set_ylabel("z (Å)");  ax1.set_title("x–z projection")
    ax2.set_xlabel("y (Å)");  ax2.set_ylabel("z (Å)");  ax2.set_title("y–z projection")
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.22, 0.5), loc='center right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Region sort key (shared utility)
# ============================================================

def region_sort_key(region):
    if region == "Above L1":
        return (0, 0)
    if region.startswith("L") and "-" in region:
        m = re.match(r"L(\d+)-L(\d+)", region)
        if m:
            return (1, int(m.group(1)))
    if region.startswith("Below"):
        return (2, 999)
    return (3, 999)


# ============================================================
# CSVs: initialise (write headers on first run)
# ============================================================

stats_header_written  = False
ml_header_written     = False

# ============================================================
# Main loop over selected snapshots
# ============================================================

all_regions_seen = set()  # accumulated across steps for consistent columns

for snap_num, snap_idx in enumerate(selected_indices, start=1):

    snapshot = all_snapshots[snap_idx]
    snapshot.wrap()

    # Try to retrieve the actual LAMMPS timestep stored in the dump.
    # ASE stores it in info["time"] (int) when reading lammps-dump-text.
    timestep = snapshot.info.get("time", snap_idx)

    step_tag = f"step_{int(timestep):010d}"

    print(
        f"\n{'='*60}",
        f"\nSnapshot {snap_num}/{len(selected_indices)}"
        f"  |  index={snap_idx}  |  timestep={timestep}",
        flush=True
    )

    # ---- Ti layer clustering --------------------------------
    ti_atoms = Atoms([a for a in snapshot if a.symbol == "Ti"])
    ti_z = ti_atoms.get_positions()[:, 2].reshape(-1, 1)

    clustering = HDBSCAN(
        copy=True,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
    ).fit(ti_z)

    labels = clustering.labels_
    n_layers_found = len(set(labels) - {-1})

    print(f"  Ti layers detected: {n_layers_found}", flush=True)

    # Save Ti layer plot
    ti_img_path = os.path.join(TI_IMG_DIR, f"ti_layers_{step_tag}.png")
    plot_clusters(snapshot, labels, ti_img_path)

    # ---- Build Delaunay surfaces ----------------------------
    layers = build_layers(snapshot, labels)

    # ---- Classify oxygens -----------------------------------
    assignments, counts = classify_oxygen_atoms(snapshot, layers)
    all_regions_seen.update(counts.keys())

    # ---- Console summary ------------------------------------
    sorted_regions = sorted(counts.keys(), key=region_sort_key)
    print(f"  {'Region':<20} {'Count':>6}")
    print(f"  {'-'*28}")
    for region in sorted_regions:
        print(f"  {region:<20} {counts[region]:>6}")
    print(f"  {'Total O':<20} {len(assignments):>6}", flush=True)

    # ---- Save oxygen region plot ----------------------------
    oxy_img_path = os.path.join(OXY_IMG_DIR, f"oxygen_regions_{step_tag}.png")
    plot_oxygen_regions(snapshot, assignments, oxy_img_path)

    # ---- Append to oxygen_statistics.csv -------------------
    stats_row = {"Timestep": timestep, "SnapshotIndex": snap_idx}
    stats_row.update(counts)
    stats_row["Total"] = len(assignments)

    stats_df = pd.DataFrame([stats_row])

    stats_df.to_csv(
        STATS_CSV,
        mode="a",
        index=False,
        header=not stats_header_written
    )
    stats_header_written = True

    # ---- ML-resolved statistics -----------------------------
    atoms_per_ml = args.atoms_per_ml
    n_ml_blocks  = len(assignments) // atoms_per_ml

    ml_rows = []
    for ml_idx in range(n_ml_blocks):
        block = assignments[ml_idx * atoms_per_ml : (ml_idx + 1) * atoms_per_ml]
        row = {
            "Timestep":     timestep,
            "SnapshotIndex": snap_idx,
            "ML":           0.25 * (ml_idx + 1),
        }
        for region in sorted_regions:
            row[region] = sum(1 for _, _, r in block if r == region)
        row["Total"] = len(block)
        ml_rows.append(row)

    if ml_rows:
        ml_df = pd.DataFrame(ml_rows)
        ml_df.to_csv(
            ML_STATS_CSV,
            mode="a",
            index=False,
            header=not ml_header_written
        )
        ml_header_written = True

    print(
        f"  Saved: {ti_img_path}\n"
        f"         {oxy_img_path}",
        flush=True
    )

# ============================================================
# Done
# ============================================================

print(f"\n{'='*60}")
print(f"Analysis complete.")
print(f"  Snapshots analyzed : {len(selected_indices)} / {total}")
print(f"  Output directory   : {os.path.abspath(OUTDIR)}")
print(f"  Overall stats CSV  : {STATS_CSV}")
print(f"  ML stats CSV       : {ML_STATS_CSV}")
print(f"  Ti layer images    : {TI_IMG_DIR}/")
print(f"  Oxygen images      : {OXY_IMG_DIR}/")