import sys
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms

# load
base = read(sys.argv[1],format='vasp')

# condition
c1 = (base.positions[:,2] > 21 )*1
c2 = (base.positions[:,2] <  1 )*1
ids = np.where( c1+c2 != 0 )[0]

# set constraint
base.set_constraint(FixAtoms(indices=ids))

# save output
write('POSCAR',images=base,format='vasp')
