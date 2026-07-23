from ase import Atoms
from ase.io import read
import numpy as np
import pandas as pd
import scipy.spatial import Delaunay
from sklearn.cluster import HDBSCAN


# PBC replication to prevent edge issue
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


# Barycentric interpolation
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


# Building ordered layers
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


