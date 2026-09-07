import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ase.io import write
import numpy as np
from ti_oxidation.rdf.rdf_optimized import run
from ti_oxidation.rdf.writer import write_rdf_log


def process_one_frame(frame, A, B, r_max, bin_size):
    return run(frame, A, B, r_max, bin_size, norm_den=True)


def start_rdf(frames, A, B, bin_size, r_max, cores):
    with ProcessPoolExecutor(max_workers=cores) as pool:
        futures = [pool.submit(
                    process_one_frame,
                    frame = frame,
                    A = A,
                    B = B,
                    r_max = r_max,
                    bin_size = bin_size
                ) for i, frame in enumerate(frames)]
        results = []
        bins = None
        failed = []
        result_norm_den = 0
        for i, f in enumerate(futures):
            try:
                res, norm_den = f.result()
                if bins is None:
                    bins = res[:,0]
                results.append(res[:,1])
                result_norm_den = result_norm_den + norm_den
            except Exception as e:
                print(f"Frame {i} failed: {e}")
                failed.append(i)
        if not results:
            raise RuntimeError("All frames failed like my CGPA")
        print(f"{len(results)}/{len(frames)} frames succeeded ({len(failed)} failed)")
        avg_gr = np.mean(np.stack(results), axis=0)
        return avg_gr, bins, result_norm_den/len(results), len(results)


def process_pickle(path, r_max, rdf_pairs, bin_size, cores=9):
    path = Path(path).resolve()
    os.chdir(path.parent)
    print(f"\n=== {path} ===")
    print(f"cwd: {os.getcwd()}")

    data = np.load(path, allow_pickle=True)
    db = data['atoms']

    queue = {
        f"rmax_{r_max}": {'db': db, 'r_max': r_max}
    }

    print("Summary:")
    for i, key in enumerate(queue):
        print(f"{i}: {len(queue[key]['db'])} frames r_max {queue[key]['r_max']}")

    for A, B in rdf_pairs:
        for key in queue:
            _db = queue[key]['db']
            _r_max = queue[key]['r_max']
            print(f"{A}-{B}: frames {len(_db)} r_max {_r_max}")
            out_path = path.parent / key / f"{A}_{B}_{_r_max}"
            out_path.mkdir(parents=True, exist_ok=True)
            gr, bins, norm_den, count = start_rdf(
                        _db,
                        A,
                        B,
                        bin_size=bin_size,
                        r_max = _r_max,
                        cores = cores
                    )
            write_rdf_log(out_path/"gr.dat", gr, bins, out_path/"log.txt", count, norm_den)


def main():
    root = Path(input("Enter directory to scan: ").strip()).resolve()
    r_max = float(input("Enter r_max: "))
    print(f"r_max {r_max}")

    all_pairs = [('Ti', 'O'), ('Ti', 'Ti'), ('O', 'O')]
    for i, (a, b) in enumerate(all_pairs):
        print(f"{i}: {a}-{b}")
    p_choices = [int(_c) for _c in (input("Enter choice: ")).strip().split()]
    rdf_pairs = [all_pairs[_c] for _c in p_choices]
    print("Pairs: ", rdf_pairs)

    bin_size = int(input("Enter bin size: "))
    print("Using bin_size: ", bin_size)

    pkls = sorted(root.rglob("*.pkl"))
    if not pkls:
        raise RuntimeError(f"No .pkl found under {root}")

    print(f"\nScanned {root}")
    print(f"Found {len(pkls)} pickles:")
    for i, p in enumerate(pkls):
        print(f"  {i}: {p}")

    raw = input("\nSelect indices (blank = all): ").strip()
    if raw:
        sel = [int(_c) for _c in raw.split()]
        pkls = [pkls[i] for i in sel]
    print(f"Selected {len(pkls)}:")
    for p in pkls:
        print(f"  {p}")
    print(f"\nr_max {r_max} | pairs {rdf_pairs} | bin_size {bin_size}")

    if input("Begin? [y/N]: ").strip().lower() not in ("y", "yes"):
        print("Aborted.")
        return

    origin = Path.cwd()
    for p in pkls:
        try:
            process_pickle(p, r_max, rdf_pairs, bin_size)
        except Exception as e:
            print(f"Pickle {p} failed: {e}")
        finally:
            os.chdir(origin)


if __name__=="__main__":
    main()