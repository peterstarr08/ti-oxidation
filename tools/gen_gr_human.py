
import pickle
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

from ti_oxidation.rdf.volume import get_effective_volume, get_effective_shell_volume

DATA_DIR = Path(r"D:\ti-oxidation\temp") / "anatase"
GR_DIR = DATA_DIR / "calculated_gr"
GR_DIR.mkdir(exist_ok=True)

TYPE_MAP = {1: "Ti", 2: "O"}
PAIRS = [("Ti", "Ti"), ("Ti", "O"), ("O", "O")]
BIN_SIZE = 200

def local_density_B(atoms, nl_matrix, ref, target, r_max):
    i, j = nl_matrix[:,0], nl_matrix[:,2]
    sel = (nl_matrix[:,1]==ref) & (nl_matrix[:,3]==target) & (i!=j) 
    nl = nl_matrix[sel]





    

     

def rdf_layer_corrected(atoms, A, B, r_max, bin_size, nl_matrix):

    # return rdf_bins(bin_size, 2), local_B_density

def process_pickle(pkl_path):
    print(f"\n=== {pkl_path.name} ===")
    with open(pkl_path, "rb") as f:
        entry = pickle.load(f)

    atoms = entry["atoms"] # cropped atoms
    nls = entry["nls"]  # {r_max: nl} with columns (i, type_i, j, type_j, d)

    for k, (r_max, nl) in enumerate(sorted(nls.items())):
        r_max = float(r_max)
        nl = np.asarray(nl)
        nl_matrix = nl
        print(f"[{k}] {len(atoms)} atoms, r_max={r_max:.3f}, "
              f"z_cut={entry['z_cut']:.3f}")

        gr = {} # Collects gr pairs Ti-Ti, Ti-O, O-O
        for A, B in PAIRS:
            pair_name = f"{A}-{B}"
            rdf_bins, rho_B = rdf_layer_corrected(
                atoms, A, B, r_max,
                bin_size=BIN_SIZE,
                nl_matrix=nl_matrix,
            )
            r_vals = np.array([r for _, r, _ in rdf_bins])
            g_vals = np.array([g for _, _, g in rdf_bins])
            gr[pair_name] = {
                "pair_symbols": (A, B),
                "rdf_bins": rdf_bins,
                "r": r_vals,
                "g": g_vals,
                "rho_B": rho_B,
                "bin_size": BIN_SIZE,
                "dr": r_max / BIN_SIZE,
                "method": "fast" if USE_FAST else "reference",
            }
            print(f"    {pair_name}: rho_B={rho_B:.6f}, "
                  f"g peak={np.nanmax(g_vals):.3f}")

        # save this (partition, r_max) result as its own pickle
        out = {
            "atoms": atoms,
            "config_file": entry["config_file"],
            "orig_indices": entry.get("orig_indices"),
            "r_max": r_max,
            "nl": nl,
            "gr": gr,
            "z_part": entry["z_part"],
            "z_cut": entry["z_cut"],
            "PART_RATIO": entry["PART_RATIO"],
            "PART_COUNT": entry["PART_COUNT"],
        }
        out_name = f"{pkl_path.stem}_rmax{r_max:.3f}_gr.pkl"
        with open(GR_DIR / out_name, "wb") as f:
            pickle.dump(out, f)
        print(f"    saved -> calculated_gr/{out_name}")


if __name__ == "__main__":
    pkls = sorted(
        p for p in DATA_DIR.glob("*.pkl")
        if p.parent == DATA_DIR  # skip anything inside calculated_gr
    )
    if not pkls:
        raise SystemExit(f"No partition pickles found in {DATA_DIR}")
    for p in pkls:
        process_pickle(p)
    print("\nAll done.")
