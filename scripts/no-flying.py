#!/usr/bin/env python3

import argparse
from ase.io import read, write
from pathlib import Path
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove atoms above a height cutoff from a structure file."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to input structure file",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        required=True,
        type=float,
        help="Height cutoff (remove atoms with z > cutoff)",
    )

    parser.add_argument(
        "-o",
        "--out",
        required=True,
        type=Path,
        help="Output file path",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    
    atoms = read(args.input, format="lammps-data")
    atoms.wrap()
    
    new_ats = atoms.copy()
   
    
    rem_index = [atom.index for atom in new_ats if atom.position[2]>=args.cutoff]
    
    del new_ats[rem_index]
            
    
    write(args.out, new_ats, format='lammps-data', atom_style='atomic', specorder=["Ti", "O"])
    
    with open(Path(args.input).parent/"info.txt", "w") as f:
        pprint(args.cutoff, stream=f)
        pprint(f'Count={len(rem_index)}', stream=f)
        pprint(rem_index, stream=f)
    

    print(f"Filtered structure written to: {args.out}")


if __name__ == "__main__":
    main()