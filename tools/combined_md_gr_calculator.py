import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ase.io import write
import numpy as np

from ti_oxidation.rdf.rdf_optimized import run 

def write_rdf(file, grs, rdf_bins):
    file = Path(file)

    with open(file, mode="w") as f:
        f.write("index,r,g(r)\n")

        for index, (r, gr) in enumerate(zip(rdf_bins, grs)):
            f.write(f"{index},{r},{gr}\n")

    print(f"File written to {file}")

def save(path, system, A, B, data):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save RDFs
    gr, bins = data["gr"]
    write_rdf(
        path / "gr.csv",
        gr,
        bins,
    )

    gr_s, bins_s = data["gr_s"]
    write_rdf(
        path / "gr_small.csv",
        gr_s,
        bins_s,
    )

    # Plain-text information
    with open(path / "info.txt", "w") as f:
        f.write(f"System              : {system}\n")
        f.write(f"Pair                : {A}-{B}\n")
        f.write(f"Number of frames    : {len(data['db'])}\n")
        f.write(f"r_max (large)       : {data['r_max_large']}\n")
        f.write(f"r_max (small)       : {data['r_max_small']}\n")
        f.write(f"r_max small index   : {data['r_max_small_indx']}\n")

    print(f"Saved information to {path / 'info.txt'}")

def process_paths(pkls):
	db = []
	for pkl in pkls:
		print(f"Process {str(pkl)}")
		data = np.load(pkl, allow_pickle=True)
		atoms = data['atoms']
		print(f"  Found {len(atoms)} frames")
		db = db + data['atoms']
	print(f"Total {len(db)} frames")
	return db

def get_all_frames():
	search_p = Path(input("Please enter search directory: ")).resolve()
	pkl_glob = list(search_p.rglob("*.pkl"))		
	print(f"Found {len(pkl_glob)}")
	return process_paths(pkl_glob)

def process_one_frame(frame, A, B, r_max, bin_size):
	return run(frame, A, B, r_max, bin_size)

def get_stat(db):
	slab_array = []
	for atoms in db:
		max_h = np.max(atoms.positions[:, 2])
		min_h = np.min(atoms.positions[:, 2])
		slab_array.append([max_h, min_h, max_h - min_h])
	slab_array = np.asarray(slab_array)  # (frames, 3)
	max_args = np.argmax(slab_array, axis=0)
	min_args = np.argmin(slab_array, axis=0)
	return {"db": db, "r_max_large": slab_array[:,2][max_args[2]], "r_max_small": slab_array[:,2][min_args[2]], "r_max_small_indx": min_args[2] }
	

def _pprint_dict(system, data_dict):
	print(f"============== {system} ==========")
	print(f"  db length           = {len(data_dict['db'])}")
	print(f"  r_max maximum       = {data_dict['r_max_large']}")
	print(f"  r_max minimum       = {data_dict['r_max_small']}")
	print(f"  r_max minimum index = {data_dict['r_max_small_indx']}")


def process_data(db_dict):
	all_data = []
	proc_db_dict = {}
	for system in db_dict:
		print(f"\nProcessing {system}")	
		_db = db_dict[system]
		_stats = get_stat(_db)
		_pprint_dict(system, _stats)
		all_data = all_data + _db
		proc_db_dict[system] = _stats
	print("\nProcessing compiled frames")
	all_stats = get_stat(all_data)
	_pprint_dict("Compiled db", all_stats)
	proc_db_dict['all_data'] = all_stats
	return proc_db_dict

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
	dbs = {}
	while(True):
		inp = input("Enter label or enter q to exit: ")
		if inp.strip()=='q':
			break
		db = get_all_frames()
		dbs[inp] = db
	data = process_data(dbs)	

	rdfs_pairs = [("Ti", "Ti"), ("Ti", "O"), ("O", "O")]

	out_path = Path(input("Enter directory to save: ")).resolve()	
	out_path.mkdir(parents=True, exist_ok=True)

	for system in data:
		print(f"Processin {system}")
		for A, B in rdfs_pairs:
			print(f"Processing {A}-{B}")
			_dict = data[system]
			_db = _dict['db']
			_rmax_l = _dict['r_max_large']
			_rmax_s = _dict['r_max_small']
			_rmax_s_indx = _dict['r_max_small_indx']
			_out = out_path / system / f"{A}_{B}"
			rdf_l, bins_l = start_rdf(_db, A, B, 1000, _rmax_l, 9)
			rdf_s, bins_s = start_rdf([_db[_rmax_s_indx]], A, B, 1000, _rmax_s, 1)
			_dict['gr'] = (rdf_l, bins_l)
			_dict['gr_s'] = (rdf_s, bins_s)

			save(_out, system, A, B, _dict)

if __name__=="__main__":
	main()
