from sklearn.cluster import HDBSCAN
from ase.io import read, write
import numpy as np

def detect_layers(atoms, filter_on='Ti', min_clust_size=192*3//4, max_clust_size=192):
    # Mask
    symbols = np.array(atoms.get_chemical_symbols())

    # Making array of indices
    filter_indxs = np.arange(len(atoms))
    z_coords = atoms.positions[:,2]
    
    # Filtering element
    sel = symbols == filter_on
    filter_indxs = filter_indxs[sel]
    z_coords = z_coords[sel].reshape(-1, 1)

    # Clustering z coordinates
    hfd = HDBSCAN(min_cluster_size=min_clust_size, max_cluster_size=max_clust_size, copy=False)
    hfd.fit(z_coords)
    labels = hfd.labels_
    # print(f"Unique labels = {np.unique(labels)}")
    
    # Collecting indices of atoms from layers
    layers = []
    for l in np.unique(labels):
        mask = labels==l
        layers.append(filter_indxs[mask])
    max_z = np.array([np.max(z_coords[indices]) for indices in layers])
    sorted_z = np.argsort(max_z)

    return [layers[l] for l in sorted_z]

    

