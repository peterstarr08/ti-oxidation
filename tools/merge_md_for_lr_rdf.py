import numpy as np
from ase.io import read
from tqdm import tqdm
from ti_oxidation.clustering.cluster_layer import detect_layers

# Flow: Take path of MD -> Display statistics of range of slab(store original statistics) --> 
# Ask for filter --> Display statistics --> Loop until satisfied --> Repeat

def delete_layers(db, layers_from_bottom):
    print("Deleting those damm layers...")
    for i in tqdm(range(len(db))):
        layers = detect_layers(db[i])
        layers = np.concatenate(layers)
        del db[i][layers]

def print_md_stats(md_stats, start_index=0):
    max_indxs = np.argmax(md_stats, axis=0) # (3,)
    min_indxs = np.argmin(md_stats, axis=0) # (3,)
    slab_h_range = np.ptp(md_stats[:,2])

    print("Slab stats:")
    print(f"\tFrames {len(md_stats)}")
    print(f"\tSlab height range {slab_h_range}")
    print(f"\t    Max height = {md_stats[:,2][max_indxs[2]]} Min height = {md_stats[:,2][min_indxs[2]]}")
    print(f'\tMax slab height {md_stats[:,2][max_indxs[2]]} Frame {start_index+max_indxs[2]}')
    print(f'\t    z_max, z_min = {md_stats[max_indxs[2]][:2]}')
    print(f'\tMax z of slab {md_stats[:,0][max_indxs[0]]} Frame {start_index+max_indxs[0]}')
    print(f'\t    z_max, z_min, slab_h = {md_stats[max_indxs[0]]}')
    print(f'\tMin z of slab {md_stats[:,1][min_indxs[1]]} Frame {start_index+min_indxs[1]}')
    print(f'\t    z_max, z_min, slab_h = {md_stats[min_indxs[1]]}')   



def get_md_stats(db):
    frames_count = len(db)
    slab_array = []
    for atoms in db:
        max_h = np.max(atoms.positions[:,2])
        min_h = np.min(atoms.positions[:,2])
        slab_array.append([max_h, min_h, max_h-min_h])
    slab_array = np.asarray(slab_array) # (frames, 3)
    return slab_array

def process_path(path):
    db = read(path, ':', format='lammps-dump-text')

    # First clearing bottom layers
    del_layers = int(input("How many layer from bottom you want to delete? "))
    delete_layers(db, del_layers)
    stats_db = get_md_stats(db)

    print_md_stats(stats_db)

    while(True):
        try:
            index1 = int(input("Great! Let's filter frames\nEnter starting index:"))
            index2 = int(input("Upto frams + 1: "))
        except Exception:
            continue
        print(f"You entered to filter {index1}:{index2}")
        inp = input("Enter y to continue:")
        if inp.strip()=='y':
            break
    print_md_stats(stats_db[index1:index2], start_index=index1)

    return db

def main():
    mds = []
    while(True):
        inp = input("Enter path or enter q to cancel\nYour input: ")
        if inp.strip()=='q':
            print("All done!")
            break
        db = process_path(inp.strip('\'').strip('\"'))
        mds = mds + db

    default_out = "./md_merged"

if __name__=="__main__":
    main()
