# POSCAR & DOSCAR for (Partial) DOS
# PROCAR for IPR (should be checked)
# try-except when file read
import sys, subprocess
import numpy as np
import matplotlib.pyplot as plt

def main():
    # User setting
    energy_rng = [-3, 3]
    plot_pdos = 1
    plot_ipr = 0
    norbit = 9 # s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2
    fs = 12
    
    # Initialization
    symbols, ntypes = read_poscar('POSCAR')
    my_dos = DOS(doscar='DOSCAR', norbit=norbit)
    my_dos.ISPIN = int(subprocess.check_output('grep ISPIN OUTCAR'.split(),universal_newlines=True).split()[2])
    energy = np.linspace(my_dos.EMIN, my_dos.EMAX, my_dos.NEDOS)
    if plot_ipr:
        my_ipr = IPR(procar='PROCAR')

    # Plotting
    fig, ax = plt.subplots()    
    # Partial DOS
    if plot_pdos:
        ATOMS = {}
        for sym, count in zip(symbols, ntypes):
            idx = sum(ntypes[:ntypes.index(count)])
            if sym not in ATOMS.keys():
                ATOMS[sym]  = list(range(idx, idx+count))
            else:
                ATOMS[sym] += list(range(idx, idx+count))
        if my_dos.ISPIN == 2:
            up   = np.arange(1,norbit*my_dos.ISPIN+1,2)
            down = np.arange(2,norbit*my_dos.ISPIN+1,2)
            for sid, sym in enumerate(symbols):
                pdos = np.sum(np.sum(my_dos.DOS[ATOMS[sym]].T[up],axis=0),axis=1)
                ax.plot(energy-my_dos.fermi, pdos)
                pdos = np.sum(np.sum(my_dos.DOS[ATOMS[sym]].T[down],axis=0),axis=1)
                ax.plot(energy-my_dos.fermi, -pdos,c=f'C{sid}', label=sym)
        else:
            up  = np.arange(1,norbit*my_dos.ISPIN+1) 
            for sid, sym in enumerate(symbols): 
                pdos = np.sum(np.sum(my_dos.DOS[ATOMS[sym]].T[up],axis=0),axis=1)
                ax.plot(energy-my_dos.fermi, pdos, c=f'C{sid}', label=sym)
            ax.set_ylim(bottom=0)
    # Total DOS
    else:
        ax.plot(energy-my_dos.fermi, my_dos.TDOS[:,1],'k')
        if my_dos.ISPIN == 2:
            ax.plot(energy-my_dos.fermi, -my_dos.TDOS[:,2],'k')
        else:
            ax.set_ylim(bottom=0)

    # Plot IPR
    if plot_ipr:
        ax_r = ax.twinx()
        ax_r.stem(my_ipr.energies - my_dos.fermi, my_ipr.ipr_values, 
                   linefmt='-', markerfmt=None, basefmt=None)
        ax_r.set_ylabel('IPR', fontsize=fs)
        ax_r.set_ylim(bottom=0)

    ax.axvline(0,c='gray', ls='--')
    ax.set_xlim(energy_rng)
    ax.set_xlabel(r'E-E$_f$ (eV)',fontsize=fs)
    ax.set_ylabel(r'DOS (a.u.)',fontsize=fs)
    ax.set_yticks([])
    ax.legend(fontsize=fs)
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------
class DOS:
    def __init__(self, doscar='DOSCAR', norbit=9):
        self.NIONS = 0
        self.EMAX, self.EMIN, self.NEDOS, self.fermi = 0, 0, 0, 0
        self.ISPIN = 1
        self.NORBIT= norbit
        self.DOS = []
        self.TDOS= []
        self.read(doscar)

    def read(self, DOSCAR):
        with open(DOSCAR,'r') as o: tmp = enumerate(o.readlines())
        self.NIONS = int(next(tmp)[1].split()[0])
        for i in range(4): next(tmp)
        self.EMAX, self.EMIN, self.NEDOS, self.fermi, _ =  map(float,next(tmp)[1].split())
        self.NEDOS = int(self.NEDOS)

        for e in range(self.NEDOS):
            _, dat = next(tmp)
            if len(self.TDOS) == 0:
                if len(dat.split()) <=3: self.ISPIN = 1
                else: self.ISPIN = 2 
                self.TDOS = np.zeros((self.NEDOS, self.ISPIN*2+1))
                self.DOS  = np.zeros((self.NIONS, self.NEDOS, self.ISPIN*self.NORBIT+1))
            else:
                self.TDOS[e] += np.array(list(map(float,dat.split())))
        n = 0
        for idx, line in tmp:
            for e in range(self.NEDOS):
                _, dat = next(tmp)
                self.DOS[n][e] += np.array(list(map(float,dat.split())))
            n+=1
        return 0

class IPR:
    def __init__(self, procar='PROCAR'):
        self.kpoints = 0
        self.bands = 0
        self.ions = 0
        self.energies = []
        self.ipr_values = []
        self.read(procar)
    
    def read(self, procar):
        with open(procar, 'r') as f:
            lines = f.readlines()
            # Read header information
            header = lines[1].split()
            self.kpoints = int(header[3])
            self.bands = int(header[7])
            self.ions = int(header[-1])
            
            data = []
            for i in range(len(lines)):
                if 'band ' in lines[i]:
                    sum_sq = 0
                    line = lines[i].split()
                    energy = float(line[4])
                    
                    # Get normalization
                    norm_line = lines[i+3+self.ions].split()
                    norm = float(norm_line[-1])
                    
                    # Calculate IPR
                    for j in range(self.ions):
                        line = lines[i+j+3].split()
                        proj_value = float(line[-1])
                        sum_sq += proj_value**2
                    
                    # Normalize IPR
                    if norm > 0:
                        ipr = sum_sq/norm**2
                    else:
                        ipr = 0
                        
                    data.append([energy, ipr])
            
            # Sort by energy and separate into arrays
            data.sort(key=lambda x: x[0])
            self.energies = np.array([x[0] for x in data])
            self.ipr_values = np.array([x[1] for x in data])

def read_poscar(poscar):
    try:
        with open(poscar, 'r') as o:
            tmp = o.readlines()
        symbols = tmp[5].split()
        ntypes = list(map(int, tmp[6].split()))
        return symbols, ntypes      
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
        print('Ions per species <-- This line is required.')
        print('Selective dynamics <-- optional')
        print(' Ion positions...')
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
