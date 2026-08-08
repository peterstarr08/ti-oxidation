"""
Logging wrapper around the existing MD slab-filtering pipeline.

This file does NOT modify the original module/functions. It imports the
existing helpers (delete_layers, get_md_stats, print_md_stats, detect_layers)
as-is and wraps the interactive workflow with exhaustive logging:

  - per-MD-file: input path, timestamp
  - layer-deletion request + before/after per-frame atom counts + before/after
    slab stats (max height, min height, ptp / slab height)
  - every filter attempt the user makes in the "pick index1:index2" loop,
    including ones they rejected (didn't confirm with 'y')
  - the final accepted filter range + resulting stats
  - a full stdout transcript (everything print_md_stats/tqdm/etc. printed)

At the end it:
  - asks for an output directory (created if missing) BEFORE any file is
    processed
  - pickles the final merged list of Atoms objects to <outdir>/md_merged.pkl
  - writes <outdir>/md_process_log.json  (structured, machine-readable)
  - writes <outdir>/md_process_log.txt   (human-readable transcript + summary)
"""

import os
import io
import sys
import json
import pickle
import datetime

import numpy as np
from ase.io import read, write
from tqdm import tqdm

# --- unchanged imports from the existing codebase ---
from ti_oxidation.clustering.cluster_layer import detect_layers


# ============================================================
# Original helpers, reproduced verbatim (NOT modified) so this
# file can run standalone. If you already have the original
# module importable, you can delete this block and instead do:
#   from your_original_module import delete_layers, get_md_stats, print_md_stats
# ============================================================
def delete_layers(db, layers_from_bottom):
    print("Deleting those damm layers...")
    print("Using first frame to detect layer...")
    layers = detect_layers(db[0])
    print(f"  Layers count {len(layers)}")
    layers_to_del = np.concatenate(layers[:layers_from_bottom])
    print(f"  Will delete {layers_from_bottom}")
    for i in range(len(db)):
        del db[i][layers_to_del]


def print_md_stats(md_stats, start_index=0):
    max_indxs = np.argmax(md_stats, axis=0)  # (3,)
    min_indxs = np.argmin(md_stats, axis=0)  # (3,)
    slab_h_range = np.ptp(md_stats[:, 2])
    print("Slab stats:")
    print(f"\tFrames {len(md_stats)}")
    print(f"\tSlab height range {slab_h_range}")
    print(f"\t    Max height = {md_stats[:,2][max_indxs[2]]} Min height = {md_stats[:,2][min_indxs[2]]}")
    print(f'\tMax slab height {md_stats[:,2][max_indxs[2]]} Frame {start_index+max_indxs[2]}')
    print(f'\t    z_max, z_min = {md_stats[max_indxs[2]][:2]}')
    print(f'\tMax z of slab {md_stats[:,0][max_indxs[0]]} Frame {start_index+max_indxs[0]}')
    print(f'\t    z_max, z_min, slab_h = {md_stats[max_indxs[0]]}')
    print(f'\tMin z of slab {md_stats[:,1][min_indxs[1]]} Frame {start_index+min_indxs[1]}')
    print(f'\t    z_max, z_min, slab_h = {md_stats[min_indxs[1]]}')


def get_md_stats(db):
    slab_array = []
    for atoms in db:
        max_h = np.max(atoms.positions[:, 2])
        min_h = np.min(atoms.positions[:, 2])
        slab_array.append([max_h, min_h, max_h - min_h])
    slab_array = np.asarray(slab_array)  # (frames, 3)
    return slab_array


# ============================================================
# Logging infrastructure
# ============================================================
class Tee(io.TextIOBase):
    """Writes to both the real stdout and an in-memory buffer, so we can
    capture every print() call (including from delete_layers/print_md_stats)
    without touching those functions."""

    def __init__(self, real_stdout, buffer):
        self.real_stdout = real_stdout
        self.buffer = buffer

    def write(self, s):
        self.real_stdout.write(s)
        self.buffer.write(s)
        return len(s)

    def flush(self):
        self.real_stdout.flush()


def _stats_summary(stats_array, start_index=0):
    """JSON-friendly numeric summary of a (frames,3) stats array
    [max_h, min_h, slab_h] per frame."""
    if len(stats_array) == 0:
        return {"n_frames": 0}
    max_indxs = np.argmax(stats_array, axis=0)
    min_indxs = np.argmin(stats_array, axis=0)
    return {
        "n_frames": int(len(stats_array)),
        "slab_height_ptp": float(np.ptp(stats_array[:, 2])),
        "max_slab_height": {
            "value": float(stats_array[max_indxs[2], 2]),
            "frame": int(start_index + max_indxs[2]),
            "z_max": float(stats_array[max_indxs[2], 0]),
            "z_min": float(stats_array[max_indxs[2], 1]),
        },
        "min_slab_height": {
            "value": float(stats_array[min_indxs[2], 2]),
            "frame": int(start_index + min_indxs[2]),
            "z_max": float(stats_array[min_indxs[2], 0]),
            "z_min": float(stats_array[min_indxs[2], 1]),
        },
        "max_z": {
            "value": float(stats_array[max_indxs[0], 0]),
            "frame": int(start_index + max_indxs[0]),
        },
        "min_z": {
            "value": float(stats_array[min_indxs[1], 1]),
            "frame": int(start_index + min_indxs[1]),
        },
    }


class MDLogger:
    def __init__(self):
        self.entries = []          # one dict per processed MD file
        self.console_transcript = []  # list of (file_index_or_None, text_chunk)
        self.merged_summary = None  # global max/min slab height across final merged db

    def new_entry(self, path):
        entry = {
            "input_path": path,
            "timestamp_start": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self.entries.append(entry)
        return entry

    def to_json(self):
        return json.dumps(
            {
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "num_files_processed": len(self.entries),
                "entries": self.entries,
                "merged_summary": self.merged_summary,
            },
            indent=2,
        )

    def to_text(self):
        lines = []
        lines.append("=" * 70)
        lines.append("MD SLAB FILTERING PIPELINE - LOG")
        lines.append(f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Files processed: {len(self.entries)}")
        lines.append("=" * 70)
        for i, e in enumerate(self.entries):
            lines.append("")
            lines.append(f"--- File {i}: {e.get('input_path')} ---")
            lines.append(f"Started: {e.get('timestamp_start')}")
            lines.append(f"Layers requested to delete from bottom: {e.get('layers_deleted')}")
            lines.append(f"Frames in file: {e.get('n_frames')}")

            b = e.get("atoms_per_frame_before")
            a = e.get("atoms_per_frame_after")
            if b is not None and a is not None:
                lines.append(f"Total atoms before deletion: {sum(b)}")
                lines.append(f"Total atoms after deletion:  {sum(a)}")
                removed = [x - y for x, y in zip(b, a)]
                lines.append(f"Atoms removed per frame (min/max/mean): "
                              f"{min(removed)}/{max(removed)}/{np.mean(removed):.2f}")

            lines.append("")
            lines.append("Stats BEFORE layer deletion:")
            lines.append(json.dumps(e.get("stats_before_deletion", {}), indent=2))
            lines.append("")
            lines.append("Stats AFTER layer deletion (full range, before frame filter):")
            lines.append(json.dumps(e.get("stats_after_deletion", {}), indent=2))

            lines.append("")
            lines.append("Filter attempts (index1, index2, user confirmation):")
            for attempt in e.get("filter_attempts", []):
                lines.append(f"  attempt {attempt['attempt_num']}: "
                              f"index1={attempt['index1']}, index2={attempt['index2']}, "
                              f"confirmed={attempt['confirmed']}")

            lines.append("")
            lines.append(f"FINAL accepted filter: "
                          f"{e.get('final_index1')}:{e.get('final_index2')}")
            lines.append("Stats for final filtered range:")
            lines.append(json.dumps(e.get("stats_final_filtered", {}), indent=2))
            lines.append(f"Finished: {e.get('timestamp_end')}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("FINAL MERGED DB SUMMARY (across all files, after all filtering)")
        lines.append("=" * 70)
        lines.append(json.dumps(self.merged_summary, indent=2))

        lines.append("")
        lines.append("=" * 70)
        lines.append("FULL CONSOLE TRANSCRIPT")
        lines.append("=" * 70)
        lines.extend(self.console_transcript)
        return "\n".join(lines)


# ============================================================
# Logged version of the interactive pipeline
# (mirrors process_path/main but adds logging; original
#  functions above are called unmodified)
# ============================================================
def process_path_logged(path, logger):
    entry = logger.new_entry(path)

    db = read(path, ':', format='lammps-dump-text')
    entry["n_frames"] = len(db)

    # --- capture BEFORE stats (prior to any layer deletion) ---
    stats_before = get_md_stats(db)
    entry["atoms_per_frame_before"] = [int(len(atoms)) for atoms in db]
    entry["stats_before_deletion"] = _stats_summary(stats_before)

    del_layers = int(input("How many layer from bottom you want to delete? "))
    entry["layers_deleted"] = del_layers

    delete_layers(db, del_layers)

    # --- capture AFTER stats ---
    stats_after = get_md_stats(db)
    entry["atoms_per_frame_after"] = [int(len(atoms)) for atoms in db]
    entry["stats_after_deletion"] = _stats_summary(stats_after)

    print_md_stats(stats_after)

    entry["filter_attempts"] = []
    attempt_num = 0
    while True:
        attempt_num += 1
        try:
            index1 = int(input("Great! Let's filter frames\nEnter starting index:"))
            index2 = int(input("Upto frams + 1: "))
        except Exception:
            entry["filter_attempts"].append({
                "attempt_num": attempt_num,
                "index1": None,
                "index2": None,
                "confirmed": False,
                "note": "invalid input (non-integer), retried",
            })
            continue
        print(f"You entered to filter {index1}:{index2}")
        inp = input("Enter y to continue:")
        confirmed = inp.strip() == 'y'
        entry["filter_attempts"].append({
            "attempt_num": attempt_num,
            "index1": index1,
            "index2": index2,
            "confirmed": confirmed,
        })
        if confirmed:
            break

    print_md_stats(stats_after[index1:index2], start_index=index1)

    entry["final_index1"] = index1
    entry["final_index2"] = index2
    entry["stats_final_filtered"] = _stats_summary(stats_after[index1:index2], start_index=index1)
    entry["timestamp_end"] = datetime.datetime.now().isoformat(timespec="seconds")

    return db[index1:index2]


def main():
    logger = MDLogger()

    # Ask for output directory FIRST, and create it.
    while True:
        out_dir = input("Enter output directory to save results/log: ").strip().strip("'").strip('"')
        if not out_dir:
            print("Please enter a non-empty path.")
            continue
        os.makedirs(out_dir, exist_ok=True)
        break
    print(f"Output directory ready: {out_dir}")

    # Tee stdout so the full transcript (including tqdm's prints,
    # print_md_stats output, etc.) is captured into the log too.
    real_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = Tee(real_stdout, buffer)

    mds = []
    try:
        while True:
            inp = input("Enter path or enter q to cancel\nYour input: ")
            if inp.strip() == 'q':
                print("All done!")
                break
            db = process_path_logged(inp.strip().strip("'").strip('"'), logger)
            mds = mds + db
    finally:
        sys.stdout = real_stdout
        logger.console_transcript = buffer.getvalue().splitlines()

    # --- compute global max/min slab height across the FINAL merged db ---
    if len(mds) > 0:
        merged_stats = get_md_stats(mds)
        max_idx = int(np.argmax(merged_stats[:, 2]))
        min_idx = int(np.argmin(merged_stats[:, 2]))
        merged_summary = {
            "n_frames_total": int(len(mds)),
            "max_slab_height": {
                "value": float(merged_stats[max_idx, 2]),
                "frame_index": max_idx,
                "z_max": float(merged_stats[max_idx, 0]),
                "z_min": float(merged_stats[max_idx, 1]),
            },
            "min_slab_height": {
                "value": float(merged_stats[min_idx, 2]),
                "frame_index": min_idx,
                "z_max": float(merged_stats[min_idx, 0]),
                "z_min": float(merged_stats[min_idx, 1]),
            },
        }
    else:
        merged_summary = {"n_frames_total": 0}

    logger.merged_summary = merged_summary

    print("Final merged db summary:")
    print(json.dumps(merged_summary, indent=2))

    # Save pickled merged Atoms db, ALONG WITH the max/min slab height summary
    merged_pkl_path = os.path.join(out_dir, "md_merged.pkl")
    with open(merged_pkl_path, "wb") as f:
        pickle.dump({"atoms": mds, "merged_summary": merged_summary}, f)
    print(f"Saved merged pickled db ({len(mds)} frames) + summary -> {merged_pkl_path}")

    # Save the merged trajectory
    merged_md_path = os.path.join(out_dir, "md_merged.extxyz")
    write(merged_md_path, mds)
    print(f"Saved merged md extexyz at {merged_md_path}")

    # Save logs
    json_path = os.path.join(out_dir, "md_process_log.json")
    txt_path = os.path.join(out_dir, "md_process_log.txt")
    with open(json_path, "w") as f:
        f.write(logger.to_json())
    with open(txt_path, "w") as f:
        f.write(logger.to_text())

    print(f"Saved structured log -> {json_path}")
    print(f"Saved human-readable log -> {txt_path}")


if __name__ == "__main__":
    main()
