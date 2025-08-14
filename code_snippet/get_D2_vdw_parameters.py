import scipy.constants as CONST
import numpy as np

# from S. Grimme, J. Comput. Chem. 27, 1787 (2007).
# https://doi.org/10.1002/jcc.20495
# J nm6 / mol
C6 = {'O': 0.70,
      'Si': 9.23}
# Angstrom
R0 = {'O': 1.342,
      'Si': 1.716}

def to_lmp_metal(pair):
    C = np.sqrt( C6[pair[0]] * C6[pair[1]] ) / CONST.e / CONST.N_A * 1E6
    R = R0[pair[0]] + R0[pair[1]]
    return np.round(C,4), np.round(R,4)

print(to_lmp_metal(['O','O']))
print(to_lmp_metal(['O','Si']))
print(to_lmp_metal(['Si','Si']))
