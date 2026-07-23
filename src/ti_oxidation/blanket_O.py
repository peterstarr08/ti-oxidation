from ase.io import read, write
from scipy.spatial import cKDTree
import numpy as np
from ase import Atom
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Blanket method to add O')
    parser.add_argument("path")
    parser.add_argument("--atoms", type=int)
    parser.add_argument("--d_min", type=float, default=1.8)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

def height_at(tree, x0, y0, xy, z, d): # (x0, y0) is a single mesh point
    indx = tree.query_ball_point([x0, y0], r=d) # detect nearest slab atoms
    # print(f'\n\n\n{xy[indx]} {x0} {y0}')
    if len(indx)==0:
        print("oh oh!")
        return -np.inf # Nothing found :(
    rho = np.sqrt(np.sum((xy[indx] - [x0, y0])**2, axis=1)) # radial distance
    # print(f'\n\n{rho} {z[indx]} {np.sqrt(d**2 - rho**2)}')
    candidates = z[indx] + np.sqrt(d**2 - rho**2)
    return candidates.max()

def main():
    args = arg_parser()
    d = args.d_min
    count = args.atoms
    atoms = read(args.path, '0', format='lammps-data')
    atoms.wrap()    # Required to prevent weird behavior when duplicatin

    
    cell = np.diag(atoms.get_cell())
    lx = cell[0]
    ly = cell[1]

    # Extracting xy and z positions

    xy = atoms.positions[:,:2]
    z = atoms.positions[:,2]

    # Repeating my shit
    shifts = np.array([[lx*x, ly*y] for x in [-1,0,1] for y in [-1,0,1]])
    rep_xy = np.concat([xy+shift for shift in shifts], axis=0)
    rep_z = np.tile(z,9)

    print(f"Shape: xy {xy.shape}\t z{z.shape}")


    # Constructnig KDtree
    print("Construcing KDTree")
    kdTree = cKDTree(rep_xy)
    print("Done")


    # Constructing meshes
    mesh_x = np.arange(0, lx, d)
    mesh_y = np.arange(0, ly, d)

    grid = np.meshgrid(mesh_x, mesh_y, indexing='ij')
    mesh_p = np.stack(grid, axis=-1).reshape(-1, 2)

    # Stuff
    heights = np.array([height_at(kdTree, x0, y0, rep_xy, rep_z, d) for x0, y0 in mesh_p])
    sample_O_p = np.column_stack([mesh_p, heights])

    # Adding O atoms
    rng = np.random.default_rng(seed=69)
    choices = rng.choice(sample_O_p, size=count, axis=0, replace=False)

    for choice in choices:
        atoms.append(Atom('O', choice))
        atoms.arrays['type'][-1] = 2

    print(f"Added {len(choices)} O atoms")
    write(args.out, atoms, format='lammps-data', specorder=['Ti', 'O'])
    print(f'Written {args.out}')
