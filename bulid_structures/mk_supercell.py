# python3.X
import sys
import numpy as np
from ase.io import read, write
from ase.build import make_supercell

align_x = 0

unit = read(sys.argv[1],format='vasp')
[xx,xy,xz] = list(map(int,sys.argv[2:]))
MAT = np.array([[xx, 0, 0],
                [0, xy, 0],
                [0, 0, xz]])
supercell = make_supercell(unit, MAT)
if align_x:
    x = np.array([1,0,0])
    a = supercell.cell[0]
    theta = np.arccos(np.clip(np.dot(a,x)/np.linalg.norm(a),-1,1))*180/np.pi
    supercell.rotate(theta, 'z', rotate_cell=True)
supercell = supercell[supercell.numbers.argsort()]
write('POSCAR_supercell.vasp', images=supercell, format='vasp')
