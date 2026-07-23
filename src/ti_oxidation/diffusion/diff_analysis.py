# Arguments
# outdir, path, every, min_cluster_size, max_cluster_size, atoms_per_ml

import os
import re

from .helper import *
from .plot import *

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


def diffusion_analysis(
            args_path,
            args_every,
            arge.min_cluster_size,
            args_max_cluster_size,
            args_atoms_per_ml
            ):


    OUTDIR          = args_outdir
    IMG_DIR         = os.path.join(OUTDIR, "images")
    TI_IMG_DIR      = os.path.join(IMG_DIR, "ti_layers")
    OXY_IMG_DIR     = os.path.join(IMG_DIR, "oxygen_regions")
    STATS_CSV       = os.path.join(OUTDIR, "oxygen_statistics.csv")
    ML_STATS_CSV    = os.path.join(OUTDIR, "oxygen_statistics_by_ml.csv")

    for d in [TI_IMG_DIR, OXY_IMG_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"Reading trajectory: {args_path}", flush=True)

    all_snapshots = read(
        args_path,
        index=":",
        format="lammps-dump-text"
    )

    total = len(all_snapshots)
    print(f"Total snapshots found: {total}", flush=True)

    # Select every Nth snapshot by index
    selected_indices = list(range(0, total, args_every))
    print(f"Snapshots to analyze:  {len(selected_indices)}  (every {args_every})", flush=True)


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
            min_cluster_size=args_min_cluster_size,
            max_cluster_size=args_max_cluster_size,
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
        atoms_per_ml = args_atoms_per_ml
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
