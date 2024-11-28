import sys
import numpy as np
from ase.io import read, write
from ase.build.supercells import make_supercell
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Parameters
L = 50
radius = 10.0
align = 'midcell'

# Functions
def merge_atoms(atom_obj, cutoff=0.1):
    global_atoms = atom_obj.get_global_number_of_atoms()
    local_atoms = global_atoms // size
    start = rank * local_atoms
    end = global_atoms if rank == size - 1 else (rank + 1) * local_atoms

    local_dist = np.zeros((end - start, global_atoms))
    local_size = local_dist.size
    sizes = comm.allgather(local_size)
    global_size = sum(sizes)
    displacements = [sum(sizes[:i]) for i in range(size)]
    global_dist = np.zeros(global_size).reshape([-1, local_dist.shape[1]])
    for i in range(start, end):
        local_dist[i-start, i:global_atoms] = atom_obj.get_distances(i, range(i, global_atoms), mic=True)
    comm.Allgatherv(sendbuf=local_dist, recvbuf=(global_dist, sizes, displacements, MPI.DOUBLE))
    global_dist += global_dist.T - np.diag(np.diag(global_dist))
    np.fill_diagonal(global_dist, np.inf)

    # Merging the overlapping atoms...
    pairs = np.argwhere( global_dist < cutoff )
    to_remove = set()
    for i, j in pairs:
        if i not in to_remove and j not in to_remove:
            to_remove.add(min(i,j))
    to_remove = np.array(list(to_remove))
    if len(to_remove) > 0:
        del atom_obj[to_remove]
    atom_obj = atom_obj[atom_obj.numbers.argsort()]
    return atom_obj

#
# make a supercell and a spherical void
umatrix = read(sys.argv[1],format='vasp')
nL = L//umatrix.cell[0][0] + 1
P = np.array([[nL, 0, 0],
              [0, nL, 0],
              [0, 0, nL]])
matrix = make_supercell(umatrix, P)
if align == 'com':
    com = matrix.get_center_of_mass()
    matrix.translate(-com)
elif align == 'midcell':
    midcell = np.sum(matrix.cell, axis=1)/2
    matrix.translate(-midcell)

ids = np.where( np.linalg.norm((matrix.positions),axis=1) < radius )[0]
del matrix[ids]
matrix = matrix[matrix.numbers.argsort()]
#write('POSCAR_matrix.vasp',images=matrix,format='vasp')

# make a spherical core structure
ucore = read(sys.argv[2], format='vasp')
nL = radius*2//ucore.cell[0][0] + 1
P = np.array([[nL, 0, 0],
              [0, nL, 0],
              [0, 0, nL]])
core = make_supercell(ucore, P)
if align == 'com':
    com = core.get_center_of_mass()
    core.translate(-com)
elif align == 'midcell':
    midcell = np.sum(core.cell, axis=1)/2
    core.translate(-midcell)

ids = np.where( np.linalg.norm((core.positions),axis=1) > radius )[0]
del core[ids]
core = core[core.numbers.argsort()]
#write('POSCAR_core.vasp',images=core,format='vasp')

# Merge two structures
my_structure = matrix + core
my_structure = my_structure[my_structure.numbers.argsort()]
my_structure.translate(np.sum(my_structure.cell, axis=1)/2)

write('output.vasp',images=my_structure,format='vasp')
