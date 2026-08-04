
import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list
from pathlib import Path

from ti_oxidation.rdf.rdf import get_avg_local_density as avg_density 
from ti_oxidation.rdf.rdf import rdf_layer_corrected as gr

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
PARTITION_RATIO = 0.2
PARTITION_COUNT = 1

def get_NL(system, cutoff):
    return np.array(neighbor_list('ij', a=system, cutoff=cutoff, self_interaction=False)).transpose()

def calculate_avg_B(system, A_name, B_name, r_max, lz, hz):
    nl = get_NL(system, r_max)
    A_indx = [atom.index for atom in system if atom.symbol==A_name]
    sum_B = 0.0
    count_A = 0
    for A in A_indx:
        position_z = system[A].position[2]
        if not (position_z >= lz and position_z <= hz):
            continue
        count_A += 1
        AB = nl[nl[:,0]==A]
        B = [int(j) for i, j in AB if system[int(j)].symbol==B_name]
        B = [indx for indx in B if A != indx]
        sum_B += len(B)
    return sum_B / count_A if count_A else float("nan")

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
        ref_bulk = atoms.copy()
        ref_bulk.pbc = [True, True, True]
        
        _atoms = atoms.copy()
#        _cell = [cell[0], cell[1], z_part]
        _atoms = [atom for atom in _atoms if atom.position[2] >= samp and atom.position[2] <= samp + z_part]
        _atoms = Atoms(_atoms, cell=atoms.get_cell())
#        _atoms.positions[:,2] -= samp
        _atoms.pbc = [True, True, False]

        del ref_bulk[[atom.index for atom in ref_bulk if atom.position[2]<samp-z_part or atom.position[2]>samp+z_part+z_part]]

        write(f"struc_x_{samp}.data", _atoms, format='lammps-data')
        write(f"struc_bulk_x_{samp}.data", ref_bulk, format='lammps-data')
        print(f"Original struc {len(atoms)} atoms. Now {len(_atoms)} size")
        z_dir.append((_atoms, 0.05, A_name, B_name, (ref_bulk, samp, samp+z_part)))
    return z_dir
    

def integrate_bins(bins):
    x = bins[:, 1]
    y = bins[:, 2]
    f = 4 * np.pi * x**2 * y
    return np.trapezoid(f, x)


def test_gr(partitions):
    for atoms, tol, A_name, B_name, ref_bulk in partitions:

        bulk_atoms, lz, hz = ref_bulk

        cell = np.diag(atoms.get_cell())
        #r_max = np.min(cell[:2])/2

        z_max = np.max(atoms.positions[:,2])
        z_min = np.min(atoms.positions[:,2])


        r_max = (z_max-z_min)/2

        B_ground = len([atom.index for atom in atoms if atom.symbol==B_name])
        B_ground_approx = B_ground / (np.prod(cell[:2])*(z_max-z_min)) * (4/3*np.pi*r_max**3)

        B_count = calculate_avg_B(bulk_atoms, A_name, B_name, r_max, lz, hz) 
        
        bins, norm_density = gr(atoms, A_name, B_name, r_max, norm_density=True)
        bins = np.array(bins)


        B_approx = norm_density*integrate_bins(bins)

        print(f"r_max {r_max}")
        print(f"Calc {B_approx}\t Actual {B_count}\tGround actual{B_ground_approx}\tRel {abs(B_approx - B_count)/B_count}")

        assert B_approx == pytest.approx(B_count, rel=tol)
        
    
