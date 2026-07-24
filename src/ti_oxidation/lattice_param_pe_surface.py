import argparse
import re
from pathlib import Path
import numpy as np

from ti_oxidation.io.lammps import *

def arg_parse():
    parser = argparse.ArgumentParser(description="Lattice Parameter Calculator using Box Method")

    parser.add_argument("path")
    parser.add_argument("--range", type=float, default=0.03)
    parser.add_argument("--steps", type=float, default=0.005)
    parser.add_argument("--lattice", nargs='+', choices=['a', 'b', 'c'])
    parser.add_argument("--override", nargs='*', type=float, default=[])
    parser.add_argument("--out-dir", default='./box_lattice_calc')

    return parser.parse_args()


def generate_input(out_dir, params, template):
    out_dir = Path(out_dir).resolve()
    param1, param2 = params
    out_path = out_dir / f'{param1[0]}_{param1[1]}_{param2[0]}_{param2[1]}'
    out_path.mkdir(parents=True, exist_ok=True)

    temp = template.update_variable(param1[0], param1[1])
    temp = temp.update_variable(param2[0], param2[1]) 

    with open(out_path/'lammps.in', 'w') as f:
        f.write(temp.content)


def main():
    args = arg_parse()
    
    if len(args.lattice) != 2:
        raise RuntimeError(f"Only 2d box search supported. Give {len(args.lattice)}")

    template = LAMMPS_Input.from_file(args.path)

    lattice_param = template.find_variable(args.lattice)

    ranges = []

    if len(args.override) != 0 and len(args.override)!=4:
        raise RuntimeError(f"Need exactly 4 param (lx1 hx1 lx2 hx2) to override")

    for i, (param, value) in enumerate(lattice_param):
        if len(args.override)==4:
            lx = args.override[2*i]
            hx = args.override[2*i+1]
            print("Overriding param for {parm}. Replaced with {lx} {hx}")
        else:
            lx = value*(1-args.range)
            hx = value*(1+args.range)
        steps = value*args.steps
        gen_range = np.arange(lx, hx, steps)
        ranges.append((param ,gen_range))
        print(f'{param}\t{value}\tlx {lx} hx {hx}\t steps {steps}\t len {len(gen_range)}')
    
    for p1_v in ranges[0][1]:
        for p2_v in ranges[1][1]:
            new_lattice = []
            new_lattice.append((ranges[0][0], p1_v))
            new_lattice.append((ranges[1][0], p2_v))
            generate_input(args.out_dir, new_lattice, template)

if __name__=="__main__":
    main()


