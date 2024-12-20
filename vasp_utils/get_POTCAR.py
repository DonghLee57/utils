# python3 POSCAR (PBE/LDA)
# Materials Project > methodology > calculation details > GGA+U calculations > Pseudo-potentials
#     https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/pseudopotentials
#
import sys, os
pot_path='your_POTCAR/PAW_PBE/'

def check_potcar_repo(dic):
    keys = list(dic.keys())
    print(f"{'POTCAR':<6s} {'Status':<8s}")
    for idx, item in enumerate(keys):
        check_path = f"{PATH_POT}/{dic[item]}"
        print(f"{item:<6s} {os.path.isdir(check_path)}")

a = {'Li': 'Li_sv', 'Na': 'Na_pv', 'K': 'K_sv', 'Cs': 'Cs_sv', 'Rb': 'Rb_sv'}
b = {'Be': 'Be_sv', 'Mg': 'Mg_pv', 'Ca': 'Ca_sv', 'Sr': 'Sr_sv', 'Ba': 'Ba_sv'}
c = {'Tc': 'Tc_pv', 'Re': 'Re_pv', 'Ru': 'Ru_pv', 'Rh': 'Rh_pv', 'Os': 'Os_pv'}
d = {'Sc': 'Sc_sv', 'Ti': 'Ti_pv',  'V': 'V_sv', 'Cr': 'Cr_pv', 'Mn': 'Mn_pv', 'Fe': 'Fe_pv', 'Ni': 'Ni_pv', 'Cu': 'Cu_pv'}
e = {'Y': 'Y_sv', 'Zr': 'Zr_sv', 'Hf': 'Hf_pv', 'Nb': 'Nb_pv', 'Ta': 'Ta_pv', 'Mo': 'Mo_pv', 'W': 'W_pv'}
f = {'Ga': 'Ga_d', 'Ge': 'Ge_d', 'In': 'In_d', 'Sn': 'Sn_d', 'Tl': 'Tl_d', 'Pb': 'Pb_d'}
g = {'Pr': 'Pr_3', 'Nd': 'Nd_3', 'Pm': 'Pm_3', 'Sm': 'Sm_3', 'Tb': 'Tb_3', 'Dy': 'Dy_3', 'Ho': 'Ho_3', 'Er': 'Er_3', 'Tm': 'Tm_3', 'Yb': 'Yb_3', 'Lu': 'Lu_3'}
mp_dictionary = {**a,**b,**c,**d,**e,**f,**g}

my_dictionanry = mp_dictionary
check_potcar_repo(my_dictionary)

try:
    with open(sys.argv[1], 'r') as o:
        types = o.readlines()[5].split()
    for i, item in enumerate(types):
        if item in my_dictionanry.keys():
            types[i] = my_dictionanry[item]
    if len(types) > 1:
        os.system('cat ' + PATH_POT + '{'+','.join(types) + '}/POTCAR > POTCAR')
    else:
        os.system('cat ' + PATH_POT + types[0] + '/POTCAR > POTCAR')
      
except IndexError:
    print('Please ensure you provide a POSCAR file in the correct format.')
    print('Command >> python (this.py) (YOUR_POSCAR)\n')
    print('##### Example of a POSCAR file format #####')
    print('Comment')
    print('Scaling factor')
    print('  Lattice[0][0] Lattice[0][1] Lattice[0][2]')
    print('  Lattice[1][0] Lattice[1][1] Lattice[1][2]')
    print('  Lattice[2][0] Lattice[2][1] Lattice[2][2]')
    print('Speicies names <-- This line is required.')
    print('Ions per species')
    print('Selective dynamics <-- optional')
    print(' Ion positions...')
