import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase import Atoms

from ti_oxidation.rdf.rdf_optimized import rdf_layer_corrected
from ti_oxidation.rdf.writer import write_rdf

def main():

    parser = argparse.ArgumentParser( description = "Calculates RDF for layer resolved system")

    parser.add_argument("path")

    parser.add_argument("--A", default='Ti')
    parser.add_argument("--B", default='O')

    parser.add_argument('--margin_top', type=float, default=0.0)
    parser.add_argument('--margin_bottom', type=float, default=0.0)

    parser.add_argument('--format', default="lammps-data")

    parser.add_argument('--nbins', type=int, default=1000)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    atoms = read(args.path, '0', format=args.format)

    input_path = Path(args.path).resolve()
    out_path = input_path.parents[0] / f'{input_path.stem}_{args.A}_{args.B}_{args.margin_top}_{args.margin_bottom}_{args.nbins}.csv'

    #filter_O = Atoms([atom for atom in atoms if atom.symbol==args.B])
    filter_O = atoms


    max_system_h = np.max(filter_O.positions[:,2]) + args.margin_top
    min_system_h = np.min(filter_O.positions[:,2]) - args.margin_bottom

    print(f"Max height {max_system_h}\t Min height {min_system_h}")

    height = max_system_h - min_system_h
    
    cell = np.diag(atoms.get_cell())

    r_max = min(cell[:2])/2         # Takes minimum of side because pbc is off in z dir and volume is corrected

    print(f'r_max {r_max}')


    # del atoms[[atom.index for atom in atoms if (atom.position[2]<min_system_h or atom.position[2]>max_system_h)]]
    atoms.pbc = [True, True, False]

    if args.debug:
        debug_path = input_path.parents[0] / f'{input_path.stem}_{args.A}_{args.B}_{args.margin_top}_{args.margin_bottom}_{args.nbins}.xyz'
        write(debug_path, atoms)
        print(f"File written {debug_path}")

    data = rdf_layer_corrected(atoms, args.A, args.B, r_max, bin_size=args.nbins)

    write_rdf(out_path, data)

