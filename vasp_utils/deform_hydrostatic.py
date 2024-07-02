from ase.io import read, write
import sys
import numpy as np

ref = read(sys.argv[1])

# Hydrostatic deformation
scaling = np.arange(0.95,1.06,0.01)
for idx, sc in enumerate(scaling):
    new = ref.copy()
    scaled = new.get_scaled_positions()
    new.cell *= sc
    new.positions = scaled @ new.cell
    #print(f"{idx+1} {new.cell[0][0]:.4f} {ref.cell[0][0]:.4f} * {sc:.2f}")
    write(f'POSCAR_{idx+1}', images=new, format='vasp')
