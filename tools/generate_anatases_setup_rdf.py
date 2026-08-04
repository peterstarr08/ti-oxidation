import pickle
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

rng = np.random.default_rng(seed=48)

# LAMMPS numeric type -> element (specorder Ti, O)
SPECORDER = ["Ti", "O"]
TYPE_MAP = {1: "Ti", 2: "O"}

anatase_p = r"D:\ti-oxidation\tests\data\anatase.data"
out_dir = Path(r"D:\ti-oxidation\temp") / "anatase"
out_dir.mkdir(exist_ok=True)

atoms = read(anatase_p, '0', format='lammps-data')


def generate_nl(atoms, r_max):
    """Build a neighbor list with ase.neighborlist.neighbor_list.

    Respects per-axis PBC from atoms.pbc (e.g. [True, True, False] slabs).
    Pure ASE convention: i is the center atom, j its neighbor, exactly as
    returned by neighbor_list('ijd', ...).
    Columns: (i, type_i, j, type_j, distance).
    """
    atoms.wrap()
    print(f"cell {atoms.get_cell()} pbc {atoms.get_pbc()}")
    print("Querying...")
    i, j, d = neighbor_list('ijd', atoms, r_max, self_interaction=False)
    print("Stacking...")

    types = atoms.arrays["type"]
    final_nl = np.column_stack((
        i,         # center index
        types[i],  # center type
        j,         # neighbor index
        types[j],  # neighbor type
        d,
    ))
    print(f"NL made! ({len(final_nl)} pairs)")
    return final_nl


def validate_nl(nl, atoms, r_max):
    """Sanity checks on the neighbor list, focused on type correctness."""
    types = np.asarray(atoms.arrays["type"])
    n_atoms = len(atoms)

    # shape / dtype
    assert nl.ndim == 2 and nl.shape[1] == 5, f"bad NL shape {nl.shape}"
    assert np.issubdtype(nl.dtype, np.floating), f"bad NL dtype {nl.dtype}"

    i, t_i, j, t_j, d = nl.T

    # index columns must be integral and in range
    for name, idx in (("i", i), ("j", j)):
        assert np.all(idx == np.round(idx)), f"{name} column not integral"
        assert idx.min() >= 0 and idx.max() < n_atoms, f"{name} index out of range"
    i = i.astype(int)
    j = j.astype(int)

    # TYPE CORRECTNESS: type columns must match types[index] exactly
    assert np.array_equal(t_i, types[i]), "center type column mismatch"
    assert np.array_equal(t_j, types[j]), "neighbor type column mismatch"

    # types themselves must be from the allowed set in the structure
    allowed = set(np.unique(types).tolist())
    assert set(np.unique(t_i)) <= allowed, "unknown center types in NL"
    assert set(np.unique(t_j)) <= allowed, "unknown neighbor types in NL"

    # no self-pairs, distances sane
    assert np.all(i != j), "self-pairs present"
    assert np.all(d > 0) and np.all(d <= r_max + 1e-8), "distances out of (0, r_max]"

    # pair symmetry: every (i, j) has a matching (j, i)
    fwd = set(zip(i.tolist(), j.tolist()))
    assert all((b, a) in fwd for a, b in fwd), "NL not symmetric"

    # type-pair symmetry: count of (t_a, t_b) bonds == count of (t_b, t_a)
    pair_counts = {}
    for a, b in zip(t_i.astype(int), t_j.astype(int)):
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    for (a, b), c in pair_counts.items():
        assert pair_counts.get((b, a), 0) == c, f"type-pair asymmetry {(a, b)}"

    print("NL validation passed "
          f"({len(nl)} pairs, types {sorted(allowed)})")


def validate_crop(cropped, original, orig_idx):
    """Check the cropped system is a faithful subset of the original.

    Ensures cropping only selected atoms and did not rewrite/reorder
    anything: types, positions, and cell must match the original system
    at the selected indices, and chemical symbols must be consistent
    with the LAMMPS types.
    """
    orig_types = np.asarray(original.arrays["type"])
    crop_types = np.asarray(cropped.arrays["type"])

    assert len(cropped) == len(orig_idx), "crop size != mask size"

    # types preserved exactly, in the same (original) order
    assert np.array_equal(crop_types, orig_types[orig_idx]), \
        "cropped types differ from original system"

    # positions untouched by the crop
    assert np.allclose(cropped.positions, original.positions[orig_idx]), \
        "cropped positions differ from original system"

    # cell carried over unchanged
    assert np.allclose(np.asarray(cropped.cell), np.asarray(original.cell)), \
        "cropped cell differs from original system"

    # symbols consistent with types (so specorder is honored on write)
    expected_symbols = [TYPE_MAP[t] for t in crop_types]
    assert list(cropped.symbols) == expected_symbols, \
        "chemical symbols inconsistent with LAMMPS types"

    # per-type counts preserved within the crop
    for t in np.unique(orig_types[orig_idx]):
        assert np.sum(crop_types == t) == np.sum(orig_types[orig_idx] == t), \
            f"type {t} count changed after crop"

    print(f"Crop validation passed ({len(cropped)} atoms, "
          f"types {sorted(set(crop_types.tolist()))})")


cell = np.diag(atoms.get_cell())
z_len = cell[2]


def generate_partitions(PARTITION_COUNT, PARTITION_RATIO):
    print(f"{PARTITION_COUNT}_{PARTITION_RATIO}")
    z_part = z_len * PARTITION_RATIO
    z_sample = rng.uniform(0, z_len - z_part, PARTITION_COUNT)
    for p_idx, samp in enumerate(z_sample):
        mask = (
            (atoms.positions[:, 2] >= samp) &
            (atoms.positions[:, 2] < samp + z_part)
        )
        orig_idx = np.flatnonzero(mask)
        _atoms = atoms[orig_idx]
        _atoms.set_cell(atoms.cell)
        _atoms.set_pbc([True, True, False])

        # assign chemical symbols from LAMMPS types so specorder is honored
        _atoms.symbols = [TYPE_MAP[t] for t in _atoms.arrays["type"]]

        # sanity check: crop must be a faithful subset of the original
        # system (types/positions/cell preserved) BEFORE exporting
        validate_crop(_atoms, atoms, orig_idx)

        # save this partition's configuration as a LAMMPS data file
        config_name = (
            f"anatase_part{p_idx}"
            f"_n{PARTITION_COUNT}_r{PARTITION_RATIO}"
            f"_z{samp:.3f}-{samp + z_part:.3f}.data"
        )
        write(
            out_dir / config_name,
            _atoms,
            format="lammps-data",
            specorder=SPECORDER,
            masses=True,
        )
        print(f"Wrote {config_name}")

        r_maxes = np.array([z_part / 2, np.min(cell[:2]) / 2]) - 1e-5
        nls = {}
        for r_max in r_maxes:
            print(f"Generating NL for {r_max}")
            nl = generate_nl(_atoms, r_max)
            validate_nl(nl, _atoms, r_max)
            nls[float(r_max)] = nl

        # export this partition separately (one pickle per partition)
        payload = {
            "atoms": _atoms,
            "config_file": config_name,
            "orig_indices": orig_idx,
            "nls": nls,          # {r_max: nl}
            "z_part": z_part,
            "z_cut": samp,
            "PART_RATIO": PARTITION_RATIO,
            "PART_COUNT": PARTITION_COUNT,
        }
        pkl_name = config_name.replace(".data", ".pkl")
        with open(out_dir / pkl_name, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Wrote {pkl_name}")


generate_partitions(5, 0.2)
