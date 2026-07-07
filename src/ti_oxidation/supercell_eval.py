from ti_oxidation.io.lammps import *
import argparse
from pathlib import Path
import io
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Supercell k parameter evaluator")

    parser.add_argument("--template")
    parser.add_argument("--out-dir")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--dry-run", action='store_true')

    args = parser.parse_args()

    # Path handling
    template_path = Path(args.template).resolve()
    out_dir = Path(args.out_dir).resolve() / f"k_{str(template_path.stem)}"

    print(f"Out dir: {out_dir}")

    # Reading template
    template = LAMMPS_Input.from_file(template_path)

    energy_list = []

    # Real shit
    for i in range(args.max_k):
        k = i + 1
        print(f"Beginning k={k}")
        
        _dir = out_dir / f"k_{k}"

        _template = template.update_variable('k', k)

        if args.dry_run:
            print(f"Would create a folder {_dir}")
            print(f"Would create a template file for k={k}")
        else:
            _dir.mkdir(parents=True, exist_ok=True)
            print(f"Created a dir at {_dir}")

            with open(_dir/"lammps.in", 'w') as f:
                f.write(_template.content)
                print(f"Lammps file written")

        # Execeution
        if args.dry_run:
            print(f"Will run lmp -in lammps.in in {_dir}")
        else:
            subprocess.run(["lmp", "-in", "lammps.in"], cwd=_dir)

        log_file = _dir / "log.lammps"
        # Obtaining log file
        if args.dry_run:
            print(f'Reading log file at {log_file}')
        else:
            results = lammps_log_extract(log_file, "k")
            energy = float(results['k'])
            print(f"Energy is {energy} eV")

            energy_list.append((k, energy))

    print(energy_list)

    run_log = out_dir/"log.out"

    if args.dry_run:
        print(f"Will write to {run_log}")
    else:
        with open(run_log, 'w') as f:
            f.write("k,energy")
            for k, energy in run_log:
                f.write(f"\n{k},{energy}")
            
        print(f"Written to {run_log}")



        


