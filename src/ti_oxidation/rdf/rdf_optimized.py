import numpy as np
from tqdm import tqdm
from ase.neighborlist import neighbor_list
import time

from .volume import get_effective_volume, get_effective_shell_volume, get_sphere_volume

def print_info(atoms, A, B, r_max, dr, bins, bin_size, method):
    print("================= RDF=================")
    print(f"Length {len(atoms)} {A}-{B} pair method {method}")
    print(f"Cell {np.diag(atoms.get_cell())} PBC {atoms.pbc}")
    print(f"Bin size {bin_size} r_max {r_max} dr {dr}")

def gen_nl(atoms, cutoff):
    _nl = neighbor_list('ijd', a=atoms, cutoff=cutoff, self_interaction=False)
    return np.asarray(_nl).transpose() # [[i,j, d],...]

def get_slab_stat(atoms):
    max_h, min_h = np.max(atoms.positions[:,2]), np.min(atoms.positions[:,2])
    slab_h = max_h - min_h
    return max_h, min_h, slab_h

def process_atom(pair_dist, bins, z_top, slab_height):
    hist,_ = np.histogram(pair_dist, bins)
    total_pairs = np.sum(hist)
    edge_l, edge_h = bins[:-1], bins[1:]

    # Corrrection for volume
    shell_volume_corr = np.asarray([get_effective_shell_volume(low, high-low, slab_height, z_top) for low, high in zip(edge_l, edge_h)])
    volume_corr = get_effective_volume(bins[-1], slab_height, z_top)

    # Fianl calculation
    local_density = total_pairs/volume_corr
    g_r = hist/shell_volume_corr

    return g_r, local_density

def process_atom_regular_gr(pair_dist, bins):
    hist,_ = np.histogram(pair_dist, bins)
    total_pairs = np.sum(hist)
    edge_l, edge_h = bins[:-1], bins[1:]

    # Volume
    shell_volume = np.asarray([get_sphere_volume(high) - get_sphere_volume(low) for low, high in zip(edge_l, edge_h)])
    volume = get_sphere_volume(bins[-1])

    # Fianl calculation
    local_density = total_pairs/volume
    g_r = hist/shell_volume

    return g_r, local_density

def rdf_layer_corrected(atoms, A, B, r_max, bin_size=200):
    # Preprocessing
    atoms.wrap()
    atoms.pbc = [True, True, False]

    max_h, min_h, slab_h = get_slab_stat(atoms)
    print(f"Slah max {max_h} min {min_h} slab_h {slab_h}")
    symbols = np.array(atoms.get_chemical_symbols())
    
    A_indices = np.where(np.array(symbols)==A)[0]
    A_z = max_h - atoms.positions[A_indices][:, 2]

    # Defining bins
    dr = r_max/bin_size
    bins = np.arange(bin_size + 1) * dr

    print_info(atoms, A, B, r_max, dr, bins, bin_size, method="Layer resolved")

    # Processing NL
    print("Generating NL")
    nl_matrix = gen_nl(atoms, r_max)
    B_mask = symbols[nl_matrix[:,1].astype(int)]==B
    nl_AB = nl_matrix[B_mask]

    # Main looper
    g_rs = []
    norm_density_sum = 0.0
    for a, z in zip(A_indices, A_z):
        a_index_nl = nl_AB[:,0] == a
        pair_dist = nl_AB[a_index_nl][:,2]
        g_r, local_density = process_atom(pair_dist, bins, z, slab_h)
        g_rs.append(g_r)
        norm_density_sum += local_density
    
    # Final calculation
    norm_density = norm_density_sum / len(A_indices)
    final_gr = np.mean(np.asarray(g_rs), axis=0) / norm_density
    return np.stack((bins[:-1], final_gr), axis=1), norm_density
    

def regular_rdf(atoms, A, B, r_max, bin_size=200):
    # Preprocessing
    atoms.wrap() #Respects system pbc
    symbols = np.array(atoms.get_chemical_symbols())
    
    A_indices = np.where(np.array(symbols)==A)[0]

    # Defining bins
    dr = r_max/bin_size
    bins = np.arange(bin_size + 1) * dr

    print_info(atoms, A, B, r_max, dr, bins, bin_size, method="Regular rdf")


    # Processing NL
    print("Generating NL")
    nl_matrix = gen_nl(atoms, r_max)
    B_mask = symbols[nl_matrix[:,1].astype(int)]==B
    nl_AB = nl_matrix[B_mask]

    # Main looper
    g_rs = []
    norm_density_sum = 0.0

    for a in tqdm(A_indices):
        a_index_nl = nl_AB[:,0] ==a
        pair_dist = nl_AB[a_index_nl][:,2]
        g_r, local_density = process_atom_regular_gr(pair_dist, bins)
        g_rs.append(g_r)
        norm_density_sum += local_density
    # Final calculation
    norm_density = norm_density_sum / len(A_indices)
    final_gr = np.mean(np.asarray(g_rs), axis=0) / norm_density
    return np.stack((bins[:-1], final_gr), axis=1), norm_density

def run(atoms, A, B, r_max, bin_size=200, regular_gr=False, norm_den=False):
    start_time = time.perf_counter()
    if regular_gr:
        print("Using regular RDF")
        print(f"PBC: {atoms.pbc}")
        result, norm = regular_rdf(atoms, A, B, r_max, bin_size)
    else:
        print("Using layer resolevd rdf")
        result, norm = rdf_layer_corrected(atoms, A, B, r_max, bin_size)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time for rdf: {execution_time:.6f} seconds")

    if norm_den:
        return result, norm
    else:
        return result
    

