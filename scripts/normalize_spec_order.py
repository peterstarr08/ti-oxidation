#!/usr/bin/env python3

from pathlib import Path
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite a LAMMPS data file with specorder=['Ti', 'O']."
    )
    parser.add_argument("input", help="Input LAMMPS data file")
    parser.add_argument(
        "--suffix",
        default="_TiO",
        help="Suffix to append to the filename (default: _TiO)",
    )
    args = parser.parse_args()

    infile = Path(args.input)

    outfile = infile.with_name(
        f"{infile.stem}{args.suffix}{infile.suffix}"
    )

    atoms = read(infile, format="lammps-data")
    
    atoms.wrap()

    write_lammps_data(
        outfile,
        atoms,
        # format='lammps-data',
        specorder=["Ti", "O"],
        masses=True,
    )

    print(f"Written: {outfile}")


if __name__ == "__main__":
    main()