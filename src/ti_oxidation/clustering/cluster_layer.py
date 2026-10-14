from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from ase.io import read, write
import numpy as np

def detect_layers(atoms, filter_on='Ti', min_clust_size=192*3//4, max_clust_size=192, cluster_selection_epsilon=0):
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
    hfd = HDBSCAN(min_cluster_size=min_clust_size, max_cluster_size=max_clust_size, copy=False, cluster_selection_epsilon=cluster_selection_epsilon)
    hfd.fit(z_coords)
    labels = hfd.labels_
    # print(f"Unique labels = {np.unique(labels)}")
 
    noise = filter_indxs[labels == -1]

    real_labels = np.unique(labels)
    real_labels = real_labels[real_labels != -1]

    layers = [filter_indxs[labels == l] for l in real_labels]
    max_z = np.array([np.max(z_coords[labels == l]) for l in real_labels])

    sorted_z = np.argsort(max_z)
    layers = [layers[i] for i in sorted_z]

    return layers, noise   




def detect_layers_dbscan(atoms, filter_on='Ti', eps=1.0, min_samples=2):
    symbols = np.array(atoms.get_chemical_symbols())
 
    filter_indxs = np.arange(len(atoms))
    z_coords = atoms.positions[:,2]
 
    sel = symbols == filter_on
    filter_indxs = filter_indxs[sel]
    z_coords = z_coords[sel].reshape(-1, 1)
 
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(z_coords)
    labels = db.labels_
 
    noise = filter_indxs[labels == -1]
 
    real_labels = np.unique(labels)
    real_labels = real_labels[real_labels != -1]
 
    layers = [filter_indxs[labels == l] for l in real_labels]
    max_z = np.array([np.max(z_coords[labels == l]) for l in real_labels])
 
    sorted_z = np.argsort(max_z)
    layers = [layers[i] for i in sorted_z]
 
    return layers, noise
 

def detect_layers_svc(atoms, n_layers, filter_on='Ti', kernel='rbf',
    C=10.0, gamma='scale', prob_threshold=0.9,
    min_layer_size=None, max_layer_size=None,
    class_weight='balanced',random_state=0):
    symbols = np.array(atoms.get_chemical_symbols())
 
    filter_indxs = np.arange(len(atoms))
    z_coords = atoms.positions[:,2]
 
    sel = symbols == filter_on
    filter_indxs = filter_indxs[sel]
    z_coords = z_coords[sel].reshape(-1, 1)
 
    z_scaled = StandardScaler().fit_transform(z_coords)
 
    # Initial labels for n_layers: split sorted z at the (n_layers - 1) largest gaps
    order = np.argsort(z_coords[:,0])
    gaps = np.diff(z_coords[order,0])
    cuts = np.sort(np.argsort(gaps)[len(gaps)-(n_layers-1):]) if n_layers > 1 else np.array([], dtype=int)
    labels = np.empty(len(z_coords), dtype=int)
    labels[order] = np.searchsorted(cuts, np.arange(len(order)), side='left')
 
    # Direct SVC fit on the gap-based labels
    svc = SVC(kernel=kernel, C=C, gamma=gamma, probability=True,
              class_weight=None, random_state=random_state)
    svc.fit(z_scaled, labels)
    proba = svc.predict_proba(z_scaled)
    labels = svc.classes_[np.argmax(proba, axis=1)]
    conf = np.max(proba, axis=1)
 
    # Probability check for noise
    noise_mask = conf < prob_threshold
 
    # Layer size constraints
    for l in range(n_layers):
        members = np.where((labels == l) & ~noise_mask)[0]
        if min_layer_size is not None and len(members) < min_layer_size:
            noise_mask[members] = True
        elif max_layer_size is not None and len(members) > max_layer_size:
            drop = members[np.argsort(conf[members])[:len(members) - max_layer_size]]
            noise_mask[drop] = True
 
    noise = filter_indxs[noise_mask]
 
    real_labels = [l for l in range(n_layers) if np.any((labels == l) & ~noise_mask)]
 
    layers = [filter_indxs[(labels == l) & ~noise_mask] for l in real_labels]
    max_z = np.array([np.max(z_coords[(labels == l) & ~noise_mask]) for l in real_labels])
 
    sorted_z = np.argsort(max_z)
    layers = [layers[i] for i in sorted_z]
 
    return layers, noise