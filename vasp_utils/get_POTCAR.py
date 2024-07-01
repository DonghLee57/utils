# python 3.X
import sys, os

PATH_POT='your_POTCAR/PAW_PBE/'
target_POSCAR = sys.argv[1]

ELE_DICT={'K':'K_pv','Ca':'Ca_pv',\
          'Sr':'Sr_sv','Y':'Y_sv','Zr':'Zr_sv','Nb':'Nb_pv',\
          'Ba':'Ba_sv'}

with open(target_POSCAR, 'r') as o: tmp = o.readlines()
types = tmp[5].split()

for i in range(len(types)):
    if types[i] in ELE_DICT.keys():
        types[i] = ELE_DICT[types[i]]

if len(types) > 1:
    os.system('cat ' + PATH_POT + '{'+','.join(types) + '}/POTCAR > POTCAR')
else:
    os.system('cat ' + PATH_POT + types[0] + '/POTCAR > POTCAR')
