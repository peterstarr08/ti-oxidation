from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ase.io import write
import numpy as np

from ti_oxidation.rdf.rdf_optimized import run 
from ti_oxidation.clustering.cluster_layer import detect_layers

def delete_layers(layers, atoms):
    del atoms[np.concatenate(layers)]

def process_one_frame(index, frame, A, B, bin_size, save_dir, regular_gr=False, use_slab_height=False, r_max=-1):
    if not regular_gr:
        print("Deleting bottom 13 layers")
        layers = detect_layers(frame)
        delete_layers(layers[:13], frame)
    else:
        print("Sparring bottom layers. Nothing was deleted!")

    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        write(path/f"{index}.extxyz", frame)
    
    # if regular_gr:
    #     r_max = np.min(np.diag(frame.get_cell()))/2
    #     if r_max > 10.0:
    #         print("Warning: System might be too large. Using 10 angs cutoff")
    #         r_max = 10.0
    # else:
    #     if use_slab_height:
    #         _max = np.max(frame.positions[:,2]) 
    #         _min = np.min(frame.positions[:,2])
    #         r_max = (_max-_min)
    #         print("Using slab height to calculate r_max")
    #     else:
    #         print("Using a and b lattice length for layer resolved rdf")
    #         r_max = np.min(np.diag(frame.get_cell())[:2])/2
    
    # print(f'Cell: {np.diag(frame.get_cell())} r_max {r_max}')
    if r_max==-1:
        _r_max = 10
        print("Temporary warning: 10 A cutoff fixed for all settings")
    else:
        _r_max = r_max
        print(f"r_max = {_r_max}")

    return run(frame, A, B, _r_max, bin_size, regular_gr)

def rdf_frames(frames, A, B, bin_size, ncores, save_dir = None, regular_gr=False, use_slab_height=False, r_max=-1):
    with ProcessPoolExecutor(max_workers=ncores) as pool:
        futures = [pool.submit(
            process_one_frame, 
            index=i, 
            frame=frame, 
            A=A, 
            B=B, 
            bin_size=bin_size, 
            save_dir=save_dir,
            regular_gr=regular_gr,
            use_slab_height=use_slab_height,
            r_max=r_max
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
            raise RuntimeError("All frames failed - no RDF computed")
     
        print(f"{len(results)}/{len(frames)} frames succeeded ({len(failed)} failed)")
        avg_gr = np.mean(np.stack(results), axis=0)

        return avg_gr, bins 
