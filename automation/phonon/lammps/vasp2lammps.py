import sys
from ase.io import read
from ase.io.lammpsdata import write_lammps_data

def read_poscar_chemical_symbols(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    element_line = lines[5].strip()
    elements = element_line.split()
    return elements

vasp = read(sys.argv[1], format='vasp')
elements = read_poscar_chemical_symbols(sys.argv[1])
print(" ".join(elements))

if len(sys.argv[1:]) == 2:
    write_lammps_data(f'{sys.argv[2]}/unit.lammps', vasp, force_skew=True)
