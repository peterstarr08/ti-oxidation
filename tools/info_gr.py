"""Inspect a calculated_gr pickle: print system info and plot the g(r) curves.

Usage:
    python inspect_gr.py
    (then paste the path to a pickle from temp/anatase/calculated_gr/)

Also accepts a raw partition pickle ({COUNT}_{RATIO}.pkl, a list of configs);
in that case it prints info for every config but there is no g(r) to plot
unless the entries contain a "gr" key.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def print_config_info(entry, label=""):
    atoms = entry["atoms"]
    pos = atoms.positions
    cell = atoms.get_cell()
    types = np.asarray(atoms.arrays["type"]).astype(int)
    unique, counts = np.unique(types, return_counts=True)

    print("=" * 60)
    if label:
        print(f"CONFIG {label}")
    print("=" * 60)
    print(f"n_atoms      : {len(atoms)}")
    print(f"composition  : " +
          ", ".join(f"type {t}: {c}" for t, c in zip(unique, counts)))
    print(f"cell         :\n{np.array(cell)}")
    print(f"cell lengths : {cell.lengths()}")
    print(f"cell angles  : {cell.angles()}")
    print(f"pbc          : {atoms.get_pbc()}")
    print(f"lowest atom z : {pos[:, 2].min():.4f}")
    print(f"highest atom z: {pos[:, 2].max():.4f}")
    print(f"slab height h : {pos[:, 2].max() - pos[:, 2].min():.4f}")
    print(f"x range      : [{pos[:, 0].min():.4f}, {pos[:, 0].max():.4f}]")
    print(f"y range      : [{pos[:, 1].min():.4f}, {pos[:, 1].max():.4f}]")

    for key in ("r_max", "z_part", "z_cut", "PART_RATIO", "PART_COUNT",
                "config_file"):
        if key in entry:
            print(f"{key:<13}: {entry[key]}")

    if "nl" in entry:
        nl = np.asarray(entry["nl"])
        print(f"NL pairs     : {len(nl)}")
        if len(nl):
            print(f"NL d range   : [{nl[:, 4].min():.4f}, {nl[:, 4].max():.4f}]")

    if "gr" in entry:
        print(f"g(r) pairs   : {list(entry['gr'].keys())}")
        for name, data in entry["gr"].items():
            g = np.asarray(data["g"])
            r = np.asarray(data["r"])
            peak = np.nanargmax(g)
            print(f"  {name:<6} rho_B={data['rho_B']:.6f}  "
                  f"dr={data['dr']:.4f}  "
                  f"first peak g={g[peak]:.3f} at r={r[peak]:.3f}")
    print("=" * 60)


def plot_gr(entry, title=""):
    gr = entry.get("gr")
    if not gr:
        print("No 'gr' key in this entry -- nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in gr.items():
        ax.plot(data["r"], data["g"], label=name, lw=1.5)

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$g_{AB}(r)$")
    ax.set_title(title or "Layer-corrected partial RDFs")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    plt.show()


def main():
    pkl_path = Path(input("Path to pickle: ").strip().strip('"').strip("'"))
    if not pkl_path.exists():
        raise SystemExit(f"Not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        # single-config pickle from calculated_gr/
        print_config_info(obj, label=pkl_path.stem)
        rmax = obj.get("r_max")
        title = f"{pkl_path.stem}" + (f"  (r_max={rmax:.3f})" if rmax else "")
        plot_gr(obj, title=title)
    elif isinstance(obj, list):
        # raw partition pickle: list of config dicts
        print(f"List pickle with {len(obj)} configs.")
        for k, entry in enumerate(obj):
            print_config_info(entry, label=f"{k} of {pkl_path.stem}")
        # plot any entries that have gr data
        plotted = 0
        for k, entry in enumerate(obj):
            if "gr" in entry:
                plot_gr(entry, title=f"{pkl_path.stem} config {k}")
                plotted += 1
        if plotted == 0:
            print("No entries contain g(r) data; run compute_layer_rdf.py "
                  "first and point this script at a calculated_gr pickle.")
    else:
        raise SystemExit(f"Unrecognized pickle content: {type(obj)}")


if __name__ == "__main__":
    main()