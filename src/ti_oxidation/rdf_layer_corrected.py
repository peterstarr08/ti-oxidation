import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase import Atoms

import argparse

from ti_oxidation.rdf.rdf_multicore import rdf_frames
from ti_oxidation.rdf.writer import write_rdf

def main():

    parser = argparse.ArgumentParser( description = "Calculates RDF for layer resolved system")

    parser.add_argument("path")

    parser.add_argument("--A", default='Ti')
    parser.add_argument("--B", default='O')
    parser.add_argument('--format', default="lammps-data")
    parser.add_argument('--nbins', type=int, default=1000)
    parser.add_argument('--debug-dir')
    parser.add_argument('--regular-gr', action='store_true') 
    parser.add_argument('--use-slab-height', '-ush', action='store_true')
    parser.add_argument('--filter', default='0')
    parser.add_argument('--out','-o', required=True)
    parser.add_argument('--r-max', type=float, default=-1)
    parser.add_argument('--cores', type=int, default=1)
    

    args = parser.parse_args()

    db = read(args.path, f'{args.filter}:', format=args.format)
    print(f"Read {len(db)} frames")
    
    input_path = Path(args.path).resolve()
    out_path = input_path.parents[0] / f'{args.out}.csv'
    
    rdf, bins = rdf_frames(db, args.A, args.B, args.nbins, args.cores, args.debug_dir, args.regular_gr, args.use_slab_height, args.r_max)

    write_rdf(out_path, rdf, bins) 


    


