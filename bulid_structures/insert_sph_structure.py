#  To be:
#    a function to delete overlap atoms at the merging step
#
import sys
import numpy as np
from ase.io import read, write
from ase.build.supercells import make_supercell

# Parameters
L = 50
radius = 10.0

#
# make a supercell and a spherical void
umatrix = read(sys.argv[1],format='vasp')
nL = L//umatrix.cell[0][0] + 1
P = np.array([[nL, 0, 0],
              [0, nL, 0],
              [0, 0, nL]])
matrix = make_supercell(umatrix, P)
com = matrix.get_center_of_mass()
matrix.translate(-com)

ids = np.where( np.linalg.norm((matrix.positions),axis=1) < radius + 1 )[0]
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
com = core.get_center_of_mass()
core.translate(-com)

ids = np.where( np.linalg.norm((core.positions),axis=1) > radius )[0]
del core[ids]
core = core[core.numbers.argsort()]
#write('POSCAR_core.vasp',images=core,format='vasp')

# Merge two structures
my_structure = matrix + core
my_structure = my_structure[my_structure.numbers.argsort()]
my_structure.translate(np.sum(my_structure.cell, axis=1)/2)

write('output.vasp',images=my_structure,format='vasp')
