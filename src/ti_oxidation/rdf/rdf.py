# Main rdf file
from ase.neighborlist import neighbor_list
import numpy as np

from .volume import get_effective_volume, get_effective_shell_volume


def get_avg_local_density(system, r_max, h, max_h, A, B, nl_matrix):
    
    density_sum = 0

    A_index = [atom.index for atom in system if atom.symbol==A]

    print(f'Found {A}\tAtoms: {len(A_index)}')
    print(f"Calculating local density of {B}")
    for i, indx in enumerate(A_index):
#        print(f"Calculating local desnity index {indx} - {i+1}/{len(A_index)}")
        # Fetch all neighbors of indx inside r_max
        AB = nl_matrix[(nl_matrix[:,0]==indx) & (nl_matrix[:,2]<=r_max)]

        # Coutn all B elements and prevent counting itself
        B_count = len([j for i,j,d in AB if (system[int(j)].symbol==B and int(j) != indx)])
        
        z = max_h - system[indx].position[2]

        eff_V = get_effective_volume(r_max, h, z)
        density_sum += B_count/eff_V

    return density_sum/len(A_index)
        

def gAB(    
            r,
            dr,
            system,
            max_h,
            h,
            A,
            B,
            nl_matrix,
            avg_local_density_B
        ):
    '''
        Cropped atoms snapshot in original box, whose layer corrected
        gAB(r) is need to be calculated
    '''

    A_index = [atom.index for atom in system if atom.symbol==A]
    
    acc_density = 0

    for indx in A_index:
        shell_AB = nl_matrix[(nl_matrix[:,0]==indx) & (nl_matrix[:,2]>=r) & (nl_matrix[:,2]<r+dr)]        
        
        count_B = len([j for i,j,d in shell_AB if (system[int(j)].symbol==B and int(j)!=indx)])
        
        z = max_h - system[indx].position[2]

        vol_shell = get_effective_shell_volume(r, dr, h, z)
    

        acc_density += count_B / (vol_shell * avg_local_density_B)

    return acc_density/len(A_index)


def rdf_layer_corrected(system, A, B, r_max, bin_size=200):

    rdf_bins = []
    dr = r_max/bin_size

    print(f"Making a cool neighbor list with {len(system)} atoms r_max {r_max}")
    NL = neighbor_list('ijd', a=system, cutoff=r_max,  self_interaction=False)
    nl_matrix = np.array(NL).transpose()

    max_h = np.max(system.positions[:,2])
    min_h = np.min(system.positions[:,2])
    
    h = max_h - min_h
    
    print(f"RDF for {A}-{B}")

    avg_local_density_B = get_avg_local_density(system, r_max, h, max_h, A, B, nl_matrix)
    print(f"Local {B} density is {avg_local_density_B}")
    
    r_values = np.arange(bin_size) * dr 
    for i, r in enumerate(r_values):
        print(f'Running g({r}) -  {i+1}/{len(r_values)}')
        rdf_bins.append(
            (i, r, gAB(r, dr, system, max_h, h, A, B, nl_matrix, avg_local_density_B))        
        )

    return rdf_bins

