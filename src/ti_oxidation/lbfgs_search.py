import argparse
import subprocess
from pathlib import Path

from ti_oxidation.minimization.bfgs import lbfgsb_optimize
from ti_oxidation.io.lammps import *

def main():
    parser = argparse.ArgumentParser(description="L-BFGS-B search for supercell")

    parser.add_argument("--template")
    parser.add_argument("--cell_param", type=float, nargs='+')
    parser.add_argument("--search", nargs='+')
    parser.add_argument("--rel-search", type=float, default=0.03)

    args = parser.parse_args()

    temp_path = Path(args.template).resolve()
    template = LAMMPS_Input.from_file(temp_path)

    work_dir = temp_path.parent / f"temp_lbfgs_{temp_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    optim_args = []
    
    lammps_arg = []
    
    for i, param in enumerate(['a', 'b', 'c']):
        _cell_param = args.cell_param[i]
        if param in args.search:
            _lb = _cell_param * (1 - args.rel_search)
            _ub = _cell_param * (1 + args.rel_search)
            
            optim_args.append((_cell_param, ( _lb, _ub)))
            lammps_arg.append((param, _cell_param))

            print(f"Search param: {param}\tGuess: {_cell_param}\tBounds: {[_lb, _ub]}")
        else:
            template = template.update_variable(param, _cell_param)
            print(f"Fixed param: {param}\tValue: {_cell_param}")

    calls = []

    def objective(guess):
        current_template = template
        variables = []
        for (param,_), x in zip(lammps_arg, guess):
            variables += [param, x]
            current_template = current_template.update_variable(param, x)

            print(f"Var={param}\tGuess={x}")
            

        lammps_path = work_dir / "lammps.in"
        log_file = work_dir / "log.lammps"

        with open(lammps_path, 'w') as f:
            f.write(current_template.content)


        subprocess.run(["lmp", "-in", "lammps.in"], cwd=work_dir, check=True)
    
        result = lammps_log_extract(log_file, ["energy"])

        energy = float(result["energy"])

        variable.append(energy)
        print(f"Result: {variable}")
        call.append(variable)
       
        return energy 

    print(calls)
    lbfgsb_optimize(objective, *optim_args)
