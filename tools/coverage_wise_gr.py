from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ase.io import write
import numpy as np

from ti_oxidation.rdf.rdf_optimized import run 
from ti_oxidation.rdf.writer import write_rdf

def process_one_frame(frame, A, B, r_max, bin_size):
	return run(frame, A, B, r_max, bin_size)

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
		for i, f in enumerate(futures):
			try:
				res = f.result()
				if bins is None:
					bins = res[:,0]
				results.append(res[:,1])
			except Exception as e:
				print(f"Frame {i} failed: {e}")
				failed.append(i)
		if not results:
			raise RuntimeError("All frames failed like my CGPA")
		print(f"{len(results)}/{len(frames)} frames succeeded ({len(failed)} failed)")
		avg_gr = np.mean(np.stack(results), axis=0)
		return avg_gr, bins

def main():
    path = Path(input("Enter path of pickle: ")).resolve()
    data = np.load(path, allow_pickle=True)
    e_r_max = None
    try:
        e_r_max = float(input("Enter optional r_max: "))
        print(f"r_max {e_r_max}")
    except Exception as e:
        print("All good, no extra calculation!")

    db = data['atoms']
    m_data = data['merged_summary']
    r_max_l = m_data['max_slab_height']['value']
    r_max_s = m_data['min_slab_height']['value']
    r_max_s_indx = m_data['min_slab_height']['frame_index']
    f_r_max_s = db[r_max_s_indx]

    # Building calculation dict
    queue = {
        'md': {'db': db, 'r_max': r_max_l},
        'rmax_small': {'db': [f_r_max_s], 'r_max':r_max_s}
    }

    if not e_r_max is None:
        queue['md_sys_r_max'] = {
                'db': db,
                'r_max': e_r_max
        }

    # Summarizing
    print("Summary:")
    for i, key in enumerate(queue):
        _len = len(queue[key]['db'])
        _r_max = queue[key]['r_max']
        print(f"{i}: {_len} frames r_max {_r_max}")

    # Calculation
    choices = input("What to do?\n").strip().split()
    choices = [int(ch) for ch in choices]

    print(f"Choices: {choices}")

    # Doing stuff
    rdf_pairs = [('Ti', 'O'), ('Ti', 'Ti'), ('O', 'O')]
    for A, B in rdf_pairs:
        for i, key in enumerate(queue):
            _db = queue[key]['db']
            _r_max = queue[key]['r_max']
            print(f"Frames {len(_db)} r_max {_r_max}")
            if not i in choices:
                print("Skipping...")
                continue
            out_path = path.parent /key /f"{A}_{B}_{_r_max}"
            out_path.mkdir(parents=True, exist_ok=True)
            gr, bins = start_rdf(
                        _db,
                        A,
                        B,
                        bin_size=1000,
                        r_max = _r_max,
                        cores = 9
                    )
            write_rdf(out_path/"gr.csv", gr, bins)



if __name__=="__main__":
    main()
