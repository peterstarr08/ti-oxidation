import argparse
import subprocess
from pathlib import Path

from ti_oxidation.minimization.bfgs import lbfgsb_optimize
from ti_oxidation.io.lammps import *

def main():
    parser = argparse.ArgumentParser(description="L-BFGS-B search for supercell")

    parser.add_argument("--template")
    parser.add_argument("--cell-param", type=float, nargs='+', required=True)
    parser.add_argument("--search", nargs='+', choices=["a","b","c"], required=True)
    parser.add_argument("--rel-search", type=float, default=0.03)

    args = parser.parse_args()

    if len(args.cell_param) != len(args.search):
        raise ValueError(f"Cell parameters count and seach should match")

    temp_path = Path(args.template).resolve()
    template = LAMMPS_Input.from_file(temp_path)

    work_dir = temp_path.parent / f"temp_lbfgs_{temp_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    optim_args = []
    
    for param, guess in zip(args.search, args.cell_param):
        _lb = guess * (1 - args.rel_search)
        _ub = guess * (1 + args.rel_search)
        
        optim_args.append((guess, ( _lb, _ub)))

        print(f"Search param: {param}\tGuess: {guess}\tBounds: {[_lb, _ub]}")

    calls = []

    def objective(guess):
        current_template = template
        variables = []
        for param, x in zip(args.search, guess):
            variables += [param, x]
            current_template = current_template.update_variable(param, x)

            print(f"Var={param}\tGuess={x}")
            

        lammps_path = work_dir / "lammps.in"
        log_file = work_dir / "log.lammps"

        with open(lammps_path, 'w') as f:
            f.write(current_template.content)


        subprocess.run(["lmp", "-in", "lammps.in"], cwd=work_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,check=True)
    
        result = lammps_log_extract(log_file, ["energy"])

        energy = float(result["energy"])

        variables.append(energy)
        print(f"Result: {variables}")
        calls.append(variables)
       
        return energy 

    lbfgsb_optimize(objective, *optim_args)
    print(calls)
