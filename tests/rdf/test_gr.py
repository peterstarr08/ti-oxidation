
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from pathlib import Path

from ti_oxidation.rdf.rdf import get_avg_local_density as avg_density 
from ti_oxidation.rdf.rdf import rdf_layer_corrected as gr

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
PARTITION_RATIO = 0.35
PARTITION_COUNT = 1

def get_NL(system, cutoff):
    return np.array(neighbor_list('ij', a=system, cutoff=cutoff, self_interaction=False)).transpose()

def calculate_avg_B(system, A_name, B_name, r_max):
    nl = get_NL(system, r_max)
    A_indx = [atom.index for atom in system if atom.symbol==A_name]
    sum_B = 0.0
    for A in A_indx:
        position = system[A].position
        #if position[2]<r_max or position[2]> np.max(system.positions[:,2]) - r_max:
        #    continue
        AB = nl[nl[:,0]==A]
        B = [int(j) for i, j in AB if system[int(j)].symbol==B_name]
        # Fail safe
        B = [indx for indx in B if A!=indx]

        sum_B += len(B)

    return sum_B / len(A_indx)

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
        _atoms.pbc = [True, True, False]
        # write(f"struc_x_{samp}.data", _atoms, format='lammps-data', specorder=['Ti', B_name])
        z_dir.append((_atoms, 0.05, A_name, B_name))
    return z_dir
    

def integrate_bins(bins):
    x = bins[:,1]
    y = bins[:,2]

    y = 4*np.pi*x*x*y

    return np.trapezoid(y, x)

def test_gr(partitions):
    for atoms, tol, A_name, B_name in partitions:
        cell = np.diag(atoms.get_cell())
        #r_max = np.min(cell[:2])/2
        r_max = np.min(cell)/2

        bins, norm_density = gr(atoms, A_name, B_name, r_max, norm_density=True)
        bins = np.array(bins)

        B_count = calculate_avg_B(atoms, A_name, B_name, r_max) 

        B_approx = norm_density*integrate_bins(bins)

        print(f"r_max {r_max}")
        print(f"Calc {B_approx}\t Actual {B_count}\tRel {abs(B_approx - B_count)/B_count}")

        assert B_approx == pytest.approx(B_count, rel=tol)
        
    
