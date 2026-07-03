import numpy as np
from ase.io import read, write
from ase import Atoms, Atom
from ase.neighborlist import neighbor_list

from ti_oxidation.rdf.rdf import get_avg_local_density as avg_density 

import pytest


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
            params=[
                    (20,20, 5, ('Ti', 92), ('O', 80), 2.5, 0.5),
                    (40,40, 20, ('Ti', 92), ('O', 80), 10, 0.5),
                    (10,20, 5, ('Ti', 50), ('O', 5), 2.5, 0.1)
                ]
        )
def get_sample(request):
    return get_structure(*request.param)

def test_local_density(get_sample):
    atoms, A, B, r_max, h, max_h, NL, volume, tolerance = get_sample

    A_name, _ = A
    B_name, B_count = B

    assert avg_density(atoms, r_max, h, max_h, A_name, B_name, NL)*volume == pytest.approx(B_count, rel=tolerance)

    
