# python3.X
import sys
import numpy as np
from ase.io import read, write
from ase.build import make_supercell

# When making a slab structure along z-axis, align_x = 1.
align_x = 0

unit = read(sys.argv[1],format='vasp')
MAT = np.array([[2, 0, 0],
                [0, 2, 0],
                [0, 0, 2]])
supercell = make_supercell(unit, MAT)

if align_x:
    x = np.array([1,0,0])
    a = supercell.cell[0]
    print(np.cross(x,a))
    theta = -np.sign(np.cross(x,a)[-1])*np.arccos(np.clip(np.dot(x,a)/np.linalg.norm(a),-1,1))*180/np.pi
    supercell.rotate(theta, 'z', rotate_cell=True)
  
supercell = supercell[supercell.numbers.argsort()]
write('POSCAR_supercell.vasp', images=supercell, format='vasp')
