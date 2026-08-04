"""Layer-resolved (layer-corrected) partial RDFs for the anatase slab partitions.

TWO METHODS, switchable with the USE_FAST bool (or the `fast=` argument
of rdf_layer_corrected):

  fast=False  ORIGINAL REFERENCE implementation, kept intact — same
              loops, same list comprehensions, same conditions, same
              scalar volume calls.

  fast=True   FAST vectorized binning that computes the SAME math:
              - density cutoff d <= r_max (same as reference)
              - self-guard j != i (same as reference)
              - bin membership via np.histogram with edges k*dr, i.e.
                the same `edge[b] <= d < edge[b+1]` comparisons as the
                reference per-bin masks
              - averages over ALL A atoms (zero-count atoms included),
                divided by len(A_index) exactly like the reference —
                no nanmean, nothing dropped
              - CORRECT VOLUME CORRECTION: get_effective_volume and
                get_effective_shell_volume are called SCALAR per atom /
                per (atom, bin), so any if/else branching inside
                volume.py is respected. No array broadcasting through
                the volume functions, ever.
              The speedup comes purely from replacing the per-atom
              per-bin re-masking of the full NL with one grouped pass.

Set VERIFY_FAST = True to run both methods on every pair and assert the
fast result matches the reference to ~1e-10 before saving.

Hard-coded NL adaptation: the saved NL from generate_partitions_ase.py
has columns (i, type_i, j, type_j, d); the reference expects nl_matrix
columns (i, j, d), so we slice nl[:, [0, 2, 4]] before passing it in.

Reads the per-partition pickles from D:\\ti-oxidation\\temp\\anatase
(anatase_part{K}_n{COUNT}_r{RATIO}_z{...}.pkl), computes g_AB(r) for
Ti-Ti, Ti-O, O-O, and saves one pickle per (partition, r_max) in
`calculated_gr`.
"""
import pickle
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

try:
    from ti_oxidation.rdf.volume import get_effective_volume, get_effective_shell_volume
except ImportError:  # running from tools/ next to the package
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ti_oxidation.rdf.volume import get_effective_volume, get_effective_shell_volume

DATA_DIR = Path(r"D:\ti-oxidation\temp") / "anatase"
GR_DIR = DATA_DIR / "calculated_gr"
GR_DIR.mkdir(exist_ok=True)

TYPE_MAP = {1: "Ti", 2: "O"}
PAIRS = [("Ti", "Ti"), ("Ti", "O"), ("O", "O")]
BIN_SIZE = 200

USE_FAST = True      # method switch: True = fast binning, False = reference
VERIFY_FAST = True  # True: run BOTH and assert they agree (slow, one-off)

# saved NL columns (ASE convention, generate_partitions_ase.py):
#   (i, type_i, j, type_j, d)
# reference nl_matrix columns: (i, j, d)
NL_TO_REFERENCE_COLS = [0, 2, 4]  # hard-coded adaptation


# ---------------------------------------------------------------------------
# REFERENCE IMPLEMENTATION — kept intact
# ---------------------------------------------------------------------------

def get_avg_local_density(system, r_max, h, max_h, A, B, nl_matrix):

    density_sum = 0
    A_index = [atom.index for atom in system if atom.symbol == A]
    print(f'Found {A}\tAtoms: {len(A_index)}')
    print(f"Calculating local density of {B}")
    for i, indx in enumerate(A_index):
#        print(f"Calculating local desnity index {indx} - {i+1}/{len(A_index)}")
        # Fetch all neighbors of indx inside r_max
        AB = nl_matrix[(nl_matrix[:, 0] == indx) & (nl_matrix[:, 2] <= r_max)]
        # Coutn all B elements and prevent counting itself
        B_count = len([j for i, j, d in AB if (system[int(j)].symbol == B and int(j) != indx)])

        z = max_h - system[indx].position[2]
        eff_V = get_effective_volume(r_max, h, z)
        density_sum += B_count / eff_V
    return density_sum / len(A_index)


def gAB(
            r,
            dr,
            system,
            max_h,
            h,
            A,
            B,
            nl_matrix,
            avg_local_density_B
        ):
    '''
        Cropped atoms snapshot in original box, whose layer corrected
        gAB(r) is need to be calculated
    '''
    A_index = [atom.index for atom in system if atom.symbol == A]

    acc_density = 0
    for indx in A_index:
        shell_AB = nl_matrix[(nl_matrix[:, 0] == indx) & (nl_matrix[:, 2] >= r) & (nl_matrix[:, 2] < r + dr)]

        count_B = len([j for i, j, d in shell_AB if (system[int(j)].symbol == B and int(j) != indx)])

        z = max_h - system[indx].position[2]
        vol_shell = get_effective_shell_volume(r, dr, h, z)

        acc_density += count_B / (vol_shell * avg_local_density_B)
    return acc_density / len(A_index)


# ---------------------------------------------------------------------------
# FAST IMPLEMENTATION — same math, vectorized counting,
# scalar (correct) volume corrections
# ---------------------------------------------------------------------------

def _counts_per_A_atom(system, nl_matrix, A_index, B, edges):
    """counts[a, b] = number of B neighbors of A atom A_index[a] with
    edges[b] <= d < edges[b+1], excluding j == i. Same comparisons as the
    reference per-bin masks (histogram edges are k*dr)."""
    symbols = np.array(system.get_chemical_symbols())
    i = nl_matrix[:, 0].astype(int)
    j = nl_matrix[:, 1].astype(int)
    d = nl_matrix[:, 2].astype(float)

    sel = (symbols[j] == B) & (j != i)
    i, d = i[sel], d[sel]         # filters only A where B atoms in j
    d_ok = d < edges[-1]          # drop only what no reference bin reaches
    i, d = i[d_ok], d[d_ok]

    local_of = np.full(len(system), -1, dtype=int) # [-1, -1, ...] size atoms total
    local_of[A_index] = np.arange(len(A_index)) # sequentially numbers A atoms 
    on_A = local_of[i] >= 0 
    i, d = i[on_A], d[on_A]

    bin_size = len(edges) - 1
    counts = np.zeros((len(A_index), bin_size))
    if len(d):
        bin_idx = np.searchsorted(edges, d, side="right") - 1  # edge[b] <= d < edge[b+1]
        np.add.at(counts, (local_of[i], bin_idx), 1.0)
    return counts


def get_avg_local_density_fast(system, r_max, h, max_h, A, B, nl_matrix):
    """Identical math to get_avg_local_density: per A atom, count B
    neighbors with d <= r_max and j != i, divide by the SCALAR effective
    volume at that atom's depth, average over ALL A atoms."""
    symbols = np.array(system.get_chemical_symbols())
    A_index = np.flatnonzero(symbols == A)
    if len(A_index) == 0:
        raise ValueError(f"no {A} atoms in this config")
    print(f'Found {A}\tAtoms: {len(A_index)}')
    print(f"Calculating local density of {B} (fast)")

    i = nl_matrix[:, 0].astype(int)
    j = nl_matrix[:, 1].astype(int)
    d = nl_matrix[:, 2].astype(float)

    sel = (symbols[j] == B) & (d <= r_max) & (j != i)
    B_count = np.zeros(len(system))
    np.add.at(B_count, i[sel], 1.0)

    density_sum = 0.0
    for indx in A_index:
        z = max_h - system.positions[indx, 2]
        eff_V = get_effective_volume(r_max, h, z)   # scalar call
        density_sum += B_count[indx] / eff_V
    return density_sum / len(A_index)


def rdf_bins_fast(system, A, B, r_max, dr, h, max_h, nl_matrix,
                  avg_local_density_B, bin_size):
    """All bins at once; per-bin math identical to gAB():
    g[b] = (1/N_A) * sum_A count_B[a, b] / (vol_shell(r_b, dr, h, z_a)
                                            * avg_local_density_B)
    with vol_shell from SCALAR calls per (atom, bin)."""
    symbols = np.array(system.get_chemical_symbols())
    A_index = np.flatnonzero(symbols == A)

    edges = np.arange(bin_size + 1) * dr
    r_values = edges[:-1]

    counts = _counts_per_A_atom(system, nl_matrix, A_index, B, edges)

    # scalar volume calls — correct branch evaluation guaranteed
    vol_shell = np.empty((len(A_index), bin_size))
    for a_loc, indx in enumerate(A_index):
        z = max_h - system.positions[indx, 2]
        for b, r in enumerate(r_values):
            vol_shell[a_loc, b] = get_effective_shell_volume(r, dr, h, z)

    # sum over ALL A atoms (zero-count atoms included), then / N_A —
    # exactly the reference average, no nanmean, nothing dropped
    per_atom = counts / (vol_shell * avg_local_density_B)
    g_values = per_atom.sum(axis=0) / len(A_index)

    return [(b, float(r_values[b]), float(g_values[b]))
            for b in range(bin_size)]


# ---------------------------------------------------------------------------
# common entry point with method switch
# ---------------------------------------------------------------------------

def rdf_layer_corrected(system, A, B, r_max, bin_size=200, debug=False,
                        norm_density=False, nl_matrix=None, fast=False):
    dr = r_max / bin_size
    # HARD-CODED CHANGE: if a ground-truth NL is supplied, use it as-is;
    # otherwise build one exactly like the reference did.
    if nl_matrix is None:
        print(f"Making a cool neighbor list with {len(system)} atoms r_max {r_max}")
        NL = neighbor_list('ijd', a=system, cutoff=r_max, self_interaction=False)
        nl_matrix = np.array(NL).transpose()
    else:
        print(f"Using saved neighbor list ({len(nl_matrix)} pairs) r_max {r_max}")
    max_h = np.max(system.positions[:, 2])
    min_h = np.min(system.positions[:, 2])

    h = max_h - min_h

    print(f"RDF for {A}-{B} [{'fast' if fast else 'reference'}]")

    if fast:
        avg_local_density_B = get_avg_local_density_fast(
            system, r_max, h, max_h, A, B, nl_matrix)
        print(f"Local {B} density is {avg_local_density_B}")
        rdf_bins = rdf_bins_fast(system, A, B, r_max, dr, h, max_h,
                                 nl_matrix, avg_local_density_B, bin_size)
    else:
        avg_local_density_B = get_avg_local_density(
            system, r_max, h, max_h, A, B, nl_matrix)
        print(f"Local {B} density is {avg_local_density_B}")
        r_values = np.arange(bin_size) * dr
        rdf_bins = []
        for i, r in enumerate(r_values):
            if debug:
                print(f'Running g({r}) -  {i+1}/{len(r_values)}')
            rdf_bins.append(
                (i, r, gAB(r, dr, system, max_h, h, A, B, nl_matrix,
                           avg_local_density_B))
            )

    if norm_density:
        return (rdf_bins, avg_local_density_B)
    return rdf_bins


def _verify_fast_vs_reference(system, A, B, r_max, nl_matrix, bin_size):
    """Run both methods and assert the fast one reproduces the reference."""
    bins_ref, rho_ref = rdf_layer_corrected(
        system, A, B, r_max, bin_size=bin_size, norm_density=True,
        nl_matrix=nl_matrix, fast=False)
    bins_fast, rho_fast = rdf_layer_corrected(
        system, A, B, r_max, bin_size=bin_size, norm_density=True,
        nl_matrix=nl_matrix, fast=True)
    g_ref = np.array([g for _, _, g in bins_ref])
    g_fast = np.array([g for _, _, g in bins_fast])
    assert np.isclose(rho_ref, rho_fast, rtol=1e-10, atol=0), \
        f"rho_B mismatch: ref={rho_ref} fast={rho_fast}"
    assert np.allclose(g_ref, g_fast, rtol=1e-9, atol=1e-12,
                       equal_nan=True), \
        f"g(r) mismatch, max |diff| = {np.nanmax(np.abs(g_ref - g_fast))}"
    print(f"VERIFY OK: fast == reference for {A}-{B} "
          f"(max |dg| = {np.nanmax(np.abs(g_ref - g_fast)):.3e})")
    return bins_fast, rho_fast


# ---------------------------------------------------------------------------
# driver for the per-partition pickles
# ---------------------------------------------------------------------------

def process_pickle(pkl_path):
    print(f"\n=== {pkl_path.name} ===")
    with open(pkl_path, "rb") as f:
        entry = pickle.load(f)

    atoms = entry["atoms"]
    nls = entry["nls"]  # {r_max: nl} with columns (i, type_i, j, type_j, d)

    for k, (r_max, nl) in enumerate(sorted(nls.items())):
        r_max = float(r_max)
        nl = np.asarray(nl)
        # hard-coded column adaptation: (i, type_i, j, type_j, d) -> (i, j, d)
        nl_matrix = nl[:, NL_TO_REFERENCE_COLS]
        print(f"[{k}] {len(atoms)} atoms, r_max={r_max:.3f}, "
              f"z_cut={entry['z_cut']:.3f}")

        gr = {}
        for A, B in PAIRS:
            pair_name = f"{A}-{B}"
            if VERIFY_FAST:
                rdf_bins, rho_B = _verify_fast_vs_reference(
                    atoms, A, B, r_max, nl_matrix, BIN_SIZE)
            else:
                rdf_bins, rho_B = rdf_layer_corrected(
                    atoms, A, B, r_max,
                    bin_size=BIN_SIZE,
                    norm_density=True,
                    nl_matrix=nl_matrix,
                    fast=USE_FAST,
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
