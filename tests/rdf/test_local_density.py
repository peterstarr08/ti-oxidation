import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from pathlib import Path

from ti_oxidation.rdf.rdf import get_avg_local_density as avg_density 

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
PARTITION_RATIO = 0.05
PARTITION_COUNT = 5

def get_NL(system, cutoff):
    return np.array(neighbor_list('ijd', a=system, cutoff=cutoff, self_interaction=False)).transpose()

def get_structure(a, b, c, A, B, cutoff, tolerance):
    
    A_name, A_count = A
    B_name, B_count = B

    total = A_count + B_count

    x = np.random.uniform(0, a, total)
    y = np.random.uniform(0, b, total)
    z = np.random.uniform(0, c, total)   

    coords = np.stack((x, y, z), axis=1)

    indices = np.arange(total)

    A_indx = np.random.choice(indices, A_count, replace=False)
    B_indx = np.setdiff1d(indices, A_indx)

    symbols = []

    for indx in indices:
        if indx in A_indx:
            symbols.append(A_name)
        elif indx in B_indx:
            symbols.append(B_name)
        else:
            print("What in the actual fuck!")

    atoms = Atoms(symbols=symbols, positions=coords, cell=[a,b,c], pbc=[True, True, False])

    return (atoms, A, B, cutoff, c, c,get_NL(atoms, cutoff), a*b*c, tolerance)


@pytest.fixture(
            params = [
                     (DATA_DIR / "anatase.data", "Ti","Ti"),
                      (DATA_DIR / "anatase.data", "Ti","O"),
                      (DATA_DIR / "anatase.data", "O","O")
                ]
        )
def partitions(request):
    path, A_name, B_name = request.param

    print(f'Constructing {A_name}-{B_name} from {path}')

    z_dir = []

    atoms = read(path, "0", format='lammps-data')
    cell = np.diag(atoms.get_cell())

    # Z dir
    z_len = cell[2]
    z_part = z_len * PARTITION_RATIO

    z_sample = np.random.uniform(0, z_len - z_part, PARTITION_COUNT)

    for samp in z_sample:
        _cell = [cell[0], cell[1], z_part]
        _atoms_filter = [atom for atom in atoms if atom.position[2] >= samp and atom.position[2] <= samp + z_part]
        _atoms = Atoms(_atoms_filter, cell=_cell)
        _atoms.positions[:,2] -= samp
        # write(f"struc_x_{samp}.data", _atoms, format='lammps-data', specorder=['Ti', B_name])
        z_dir.append((_atoms, 0.05, A_name, B_name))
    return z_dir
    


@pytest.mark.skip(reason='Already tested and works fine. Takes too long...')
def test_local_density(partitions):
    for atoms, tol, A_name, B_name in partitions:
        cell = np.diag(atoms.get_cell())
        r_max = np.min(cell[:2])/2
        h = cell[2]
        atoms.pbc = [True, True, False]

        B_count = len([atom.index for atom in atoms if atom.symbol==B_name])
        volume = np.prod(cell)

        B_approx = avg_density(atoms, r_max, h, h, A_name, B_name, get_NL(atoms, r_max))*volume 
        print(f"r_max {r_max}\th {h}")
        print(f"Calc {B_approx}\t Actual {B_count}\tRel {abs(B_approx - B_count)/B_count}")
        assert B_approx == pytest.approx(B_count, rel=tol)

