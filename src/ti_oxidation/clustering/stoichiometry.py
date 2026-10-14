from .cluster_layer import detect_layers, detect_layers_dbscan, detect_layers_svc
import numpy as np

def detect_ti_o_layers_hdbscan(atoms, ti_min=50, ti_max=200, o_min=10, o_max=800):
    ti_layers, ti_noise = detect_layers(atoms, 'Ti', min_clust_size=ti_min, max_clust_size=ti_max,cluster_selection_epsilon=0.2)
    o_layers, o_noise = detect_layers(atoms, 'O', min_clust_size=o_min, max_clust_size=o_max, cluster_selection_epsilon=0.2)

    z_pos = atoms.positions[:,2]
    ti_mean_p_l = np.array([z_pos[l].mean() for l in ti_layers])
    o_mean_p_l  = np.array([z_pos[l].mean() for l in o_layers])

    layers = ti_layers + o_layers
    mean_layers = np.concatenate((ti_mean_p_l, o_mean_p_l))
    n_ti = len(ti_mean_p_l)

    comb_ind = np.argsort(mean_layers)

    ti_count = 0
    o_count = 0
    tagged_layers = []
    for cind in comb_ind:
        if cind < n_ti:
            tag, count = 'Ti', ti_count
            ti_count += 1
        else:
            tag, count = 'O', o_count
            o_count += 1
        tagged_layers.append((tag, count, layers[cind]))

    return tagged_layers, ti_noise, o_noise



def detect_ti_o_layers_dbscan(atoms, ti_eps=1.0, ti_min_samples=2, o_eps=1.0, o_min_samples=2):
    ti_layers, ti_noise = detect_layers_dbscan(atoms, 'Ti', eps=ti_eps, min_samples=ti_min_samples)
    o_layers, o_noise = detect_layers_dbscan(atoms, 'O', eps=o_eps, min_samples=o_min_samples)
 
    z_pos = atoms.positions[:,2]
    ti_mean_p_l = np.array([z_pos[l].mean() for l in ti_layers])
    o_mean_p_l  = np.array([z_pos[l].mean() for l in o_layers])
 
    layers = ti_layers + o_layers
    mean_layers = np.concatenate((ti_mean_p_l, o_mean_p_l))
    n_ti = len(ti_mean_p_l)
 
    comb_ind = np.argsort(mean_layers)
 
    ti_count = 0
    o_count = 0
    tagged_layers = []
    for cind in comb_ind:
        if cind < n_ti:
            tag, count = 'Ti', ti_count
            ti_count += 1
        else:
            tag, count = 'O', o_count
            o_count += 1
        tagged_layers.append((tag, count, layers[cind]))
 
    return tagged_layers, ti_noise, o_noise



def detect_ti_o_layers_svc(atoms, n_ti_layers, n_o_layers, ti_kernel='rbf', ti_C=10.0, ti_gamma='scale',
                           o_kernel='rbf', o_C=10.0, o_gamma='scale', ti_prob_threshold=0.9, o_prob_threshold=0.9,
                           ti_min_layer_size=None, ti_max_layer_size=None, o_min_layer_size=None, o_max_layer_size=None,
                           class_weight='balanced', random_state=0):
    ti_layers, ti_noise = detect_layers_svc(atoms, n_ti_layers, filter_on='Ti', kernel=ti_kernel, C=ti_C, gamma=ti_gamma,
                                            prob_threshold=ti_prob_threshold, min_layer_size=ti_min_layer_size,
                                            max_layer_size=ti_max_layer_size, class_weight=class_weight, random_state=random_state)
    o_layers, o_noise = detect_layers_svc(atoms, n_o_layers, filter_on='O', kernel=o_kernel, C=o_C, gamma=o_gamma,
                                          prob_threshold=o_prob_threshold, min_layer_size=o_min_layer_size,
                                          max_layer_size=o_max_layer_size, class_weight=class_weight, random_state=random_state)
 
    z_pos = atoms.positions[:,2]
    ti_mean_p_l = np.array([z_pos[l].mean() for l in ti_layers])
    o_mean_p_l  = np.array([z_pos[l].mean() for l in o_layers])
 
    layers = ti_layers + o_layers
    mean_layers = np.concatenate((ti_mean_p_l, o_mean_p_l))
    n_ti = len(ti_mean_p_l)
 
    comb_ind = np.argsort(mean_layers)
 
    ti_count = 0
    o_count = 0
    tagged_layers = []
    for cind in comb_ind:
        if cind < n_ti:
            tag, count = 'Ti', ti_count
            ti_count += 1
        else:
            tag, count = 'O', o_count
            o_count += 1
        tagged_layers.append((tag, count, layers[cind]))
 
    return tagged_layers, ti_noise, o_noise