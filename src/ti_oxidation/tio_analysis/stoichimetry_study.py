"""
trajectory_layer_clustering.py

Detects Ti/O atomic layers along the z-axis for every frame of a LAMMPS
dump trajectory, saves debug plots for ~10 evenly spaced frames, and
stores per-frame clustering results in a single .npz file.

NPZ FILE FORMAT (layer_data.npz)
=================================
frame_numbers   (F,) int64    trajectory frame index, one entry per processed frame

layer_frame     (L,) int64    index into frame_numbers this layer belongs to
layer_tag       (L,) '<U2'    'Ti' or 'O'
layer_count     (L,) int64    0-based index of this layer within its own element
                               (Ti0, Ti1, ..., O0, O1, ...), bottom to top
layer_zmean     (L,) float64  mean z coordinate of atoms in this layer
layer_start     (L,) int64    start offset into layer_atoms for this layer
layer_end       (L,) int64    end offset (exclusive) into layer_atoms

layer_atoms     (N,) int64    concatenated atom indices for all layers, all frames
                               layer i's atom indices = layer_atoms[layer_start[i]:layer_end[i]]

noise_frame     (M,) int64    index into frame_numbers this noise entry belongs to
noise_tag       (M,) '<U2'    'Ti' or 'O'
noise_start     (M,) int64    start offset into noise_atoms
noise_end       (M,) int64    end offset (exclusive) into noise_atoms

noise_atoms     (K,) int64    concatenated noise atom indices, all frames
                               noise entry j's atom indices = noise_atoms[noise_start[j]:noise_end[j]]

Each frame has exactly 2 noise entries (one 'Ti', one 'O'), possibly empty.

Example reconstruction for one frame:
    data = np.load('layer_data.npz')
    f = 0  # position in frame_numbers, not the raw trajectory frame number
    for i in np.where(data['layer_frame'] == f)[0]:
        idx = data['layer_atoms'][data['layer_start'][i]:data['layer_end'][i]]
        tag, count = data['layer_tag'][i], data['layer_count'][i]
"""

import os
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ase.io
from tqdm import tqdm

from ti_oxidation.clustering.stoichiometry import (
    detect_ti_o_layers_hdbscan,
    detect_ti_o_layers_dbscan,
    detect_ti_o_layers_svc,
)

MARKERS = ['s', '^', 'D', 'v', 'P', '*', 'X', 'h']

# Output resolution for saved debug plots. Figure size and font sizes come
# from whatever matplotlib style is active (e.g. an ACS style set elsewhere
# in the pipeline), rather than being hard-coded here.
_SAVE_DPI = 300

DETECTORS = {
    'hdbscan': detect_ti_o_layers_hdbscan,
    'dbscan': detect_ti_o_layers_dbscan,
    'svc': detect_ti_o_layers_svc,
}


def plot_frame(atoms, tagged_layers, ti_noise, o_noise, save_path):
    pos = atoms.positions
    plt.figure()

    for tag, count, idx in tagged_layers:
        p = pos[idx]
        if tag == 'Ti':
            plt.scatter(p[:, 0], p[:, 2], s=2, marker='o', label=f'Ti{count}')
        else:
            plt.scatter(p[:, 0], p[:, 2], s=8, marker=MARKERS[count % len(MARKERS)], label=f'O{count}')

    p = pos[ti_noise]
    plt.scatter(p[:, 0], p[:, 2], s=40, marker='x', c='red', linewidths=2, label='Ti noise')

    p = pos[o_noise]
    plt.scatter(p[:, 0], p[:, 2], s=40, marker='+', c='red', linewidths=2, label='O noise')

    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend(markerscale=1, ncol=1, fontsize=6, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(save_path, dpi=_SAVE_DPI, bbox_inches='tight')
    plt.close()


def process_frame(args):
    frame_idx, atoms, method, method_kwargs, plot_path = args
    detect_fn = DETECTORS[method]
    tagged_layers, ti_noise, o_noise = detect_fn(atoms, **method_kwargs)

    if plot_path is not None:
        plot_frame(atoms, tagged_layers, ti_noise, o_noise, plot_path)

    return frame_idx, atoms, tagged_layers, ti_noise, o_noise


def cluster_trajectory(traj_path, out_dir, method='hdbscan', method_kwargs=None,
                        n_debug_plots=10, n_workers=9):
    if method not in DETECTORS:
        raise ValueError(f'unknown method: {method}, expected one of {list(DETECTORS)}')

    method_kwargs = method_kwargs or {}

    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, 'debug_plots')
    os.makedirs(debug_dir, exist_ok=True)

    print(f'reading trajectory: {traj_path}')
    frames = ase.io.read(traj_path, format='lammps-dump-text', index=':')
    n_frames = len(frames)

    debug_idx = set(np.linspace(0, n_frames - 1, min(n_debug_plots, n_frames), dtype=int).tolist())

    print(f'frames found        : {n_frames}')
    print(f'method              : {method}')
    print(f'method kwargs       : {method_kwargs}')
    print(f'debug plots         : {len(debug_idx)} -> {debug_dir}')
    print(f'workers             : {n_workers}')
    print(f'output npz          : {os.path.join(out_dir, "layer_data.npz")}')

    tasks = []
    for i, atoms in enumerate(frames):
        plot_path = os.path.join(debug_dir, f'frame_{i:05d}.png') if i in debug_idx else None
        tasks.append((i, atoms, method, method_kwargs, plot_path))

    results = [None] * n_frames
    with mp.Pool(processes=n_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_frame, tasks), total=n_frames, desc='clustering frames'):
            results[res[0]] = res

    frame_numbers = []
    layer_frame, layer_tag, layer_count, layer_zmean = [], [], [], []
    layer_start, layer_end = [], []
    layer_atoms = []

    noise_frame, noise_tag, noise_start, noise_end = [], [], [], []
    noise_atoms = []

    print('aggregating results and writing npz...')
    for frame_idx, atoms, tagged_layers, ti_noise, o_noise in results:
        f_pos = len(frame_numbers)
        frame_numbers.append(frame_idx)
        z_pos = atoms.positions[:, 2]

        for tag, count, idx in tagged_layers:
            start = len(layer_atoms)
            layer_atoms.extend(idx.tolist())
            end = len(layer_atoms)

            layer_frame.append(f_pos)
            layer_tag.append(tag)
            layer_count.append(count)
            layer_zmean.append(z_pos[idx].mean())
            layer_start.append(start)
            layer_end.append(end)

        for tag, idx in (('Ti', ti_noise), ('O', o_noise)):
            start = len(noise_atoms)
            noise_atoms.extend(idx.tolist())
            end = len(noise_atoms)

            noise_frame.append(f_pos)
            noise_tag.append(tag)
            noise_start.append(start)
            noise_end.append(end)

    out_path = os.path.join(out_dir, 'layer_data.npz')
    np.savez(
        out_path,
        frame_numbers=np.array(frame_numbers, dtype=np.int64),
        layer_frame=np.array(layer_frame, dtype=np.int64),
        layer_tag=np.array(layer_tag, dtype='<U2'),
        layer_count=np.array(layer_count, dtype=np.int64),
        layer_zmean=np.array(layer_zmean, dtype=np.float64),
        layer_start=np.array(layer_start, dtype=np.int64),
        layer_end=np.array(layer_end, dtype=np.int64),
        layer_atoms=np.array(layer_atoms, dtype=np.int64),
        noise_frame=np.array(noise_frame, dtype=np.int64),
        noise_tag=np.array(noise_tag, dtype='<U2'),
        noise_start=np.array(noise_start, dtype=np.int64),
        noise_end=np.array(noise_end, dtype=np.int64),
        noise_atoms=np.array(noise_atoms, dtype=np.int64),
    )

    print(f'done: {out_path}')
    return out_path


if __name__ == '__main__':
    import sys
    cluster_trajectory(sys.argv[1], sys.argv[2])