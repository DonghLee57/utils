# Read POSCAR, OUTCAR, DOSCAR
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    # setting
    energy_rng = [-2, 2]
    plot_pdos = 0
    norbit = 9 # s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2
  
    # initialization
    symbols, ntypes = read_poscar('POSCAR')
    my_dos = DOS(doscar='DOSCAR', norbit=norbit)

    # Total DOS
    fig, ax = plt.subplots()
    energy = np.linspace(my_dos.EMIN, my_dos.EMAX, my_dos.NEDOS)
    ax.plot(energy-my_dos.fermi, my_dos.TDOS[:,1],'k')
    if my_dos.ISPIN == 2:
        ax.plot(energy-my_dos.fermi, -my_dos.TDOS[:,2],'k')
    else:
        ax.set_ylim(bottom=0)

    # Partial DOS (There maybe bugs...)
    if plot_pdos:
        PDOS = {}
        ATOMS = {}
        for e in ntypes:
            ATOMS[e] = [] 
            for i, item in enumerate(symbols):
                if e==item:
                    ATOMS[e].append(i)

        if my_dos.ISPIN == 2:
            ORBIT  = np.arange(1,norbit*my_dos.ISPIN+1,2)
            ORBIT2 = np.arange(2,norbit*my_dos.ISPIN+1,2)
            for e in ntypes:
                PDOS[e] = np.sum(np.sum(my_dos.DOS[ATOMS[e]].T[ORBIT],axis=0),axis=1)
                ax.plot(energy-my_dos.fermi, PDOS[e])
            for e in ntypes:
                PDOS[e] = np.sum(np.sum(my_dos.DOS[ATOMS[e]].T[ORBIT2],axis=0),axis=1)
                ax.plot(energy-my_dos.fermi, -PDOS[e],c='C'+str(ntypes.index(e)))
        else:
            ORBIT  = np.arange(1,norbit*my_dos.ISPIN+1) 
            for e in ntypes:
                PDOS[e] = np.zeros(my_dos.NEDOS) 
                for i in range(len(ATOMS[e])):
                    PDOS[e] = np.sum(my_dos.DOS[ATOMS[e]].T[ORBIT,],axis=0)
                ax.plot(energy-my_dos.fermi, PDOS[e],c='C'+str(c_index.index(e)))
              
    ax.axvline(0,c='gray', ls='--')
    ax.set_xlim(energy_rng)
    ax.set_xlabel(r'E-E$_f$ (eV)',fontsize=12)
    ax.set_ylabel(r'DOS (a.u.)',fontsize=12)
    ax.set_yticks([])
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
                self.TDOS = np.zeros((self.NEDOS,self.ISPIN*2+1))
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

def read_poscar(poscar):
    try:
        with open(poscar, 'r') as o:
            symbols = o.readlines()[5].split()
            ntypes = o.readlines()[6].split()
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
