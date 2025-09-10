import sys
from ase.data import atomic_numbers, atomic_masses

input_elem = sys.argv[1:]
elements = input_elem.copy()
elements.sort(key=lambda x: atomic_numbers[x])

print('pair_coeff       * * ${NMPL} ${POT} ' + f'{" ".join(elements)}\n')

for idx, elem in enumerate(input_elem):
    print(f"mass\t{idx+1} {atomic_masses[atomic_numbers[elem]]:.4f}")
