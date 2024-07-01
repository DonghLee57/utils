import numpy as np
import ase
from ase.io import read, write
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import sys
def main():
    my = StructureAnalysis(sys.argv[1], 'vasp')

    rdf, cn_distribution = my.calculate_rdf(6, 2.5, 0.005)
    np.savetxt('rdf.out', rdf, fmt='%.4f')
    np.savetxt('cn_distribution.out', cn_distribution, fmt='%.4f')

    return 0

class StructureAnalysis:
    def __init__(self, filename:str, fileformat:str, index='-1'):
        self.structure = read(filename, index=index, format=fileformat)
        self.cn_lim = [0, 10]

    def calculate_single_rdf(self, atoms:ase.atom.Atom, rmax:float, cutoff:float, dr:float):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        rdf = np.zeros(len(bins) - 1)
        if rmax > atoms.get_cell().diagonal().min() / 2:
            print('WARNING: The input maximum radius is over the half the smallest cell dimension.')
        nions = atoms.get_global_number_of_atoms()
        local_nions = nions // size
        start = rank * local_nions
        end = nions if rank == size - 1 else (rank + 1) * local_nions
        local_dist = np.zeros((end - start, nions))
        for i in range(start, end):
            distances = atoms.get_distances(i, range(i, nions), mic=True)
            local_dist[i - start, i:nions] = distances
        full_dist = np.zeros((nions, nions))
        comm.Allgatherv(local_dist, full_dist)
        full_dist += full_dist.T - np.diag(np.diag(full_dist))
        np.fill_diagonal(full_dist, np.inf)
        res, bin_edges = np.histogram(full_dist, bins=bins)
        rdf += res / ((nions ** 2 / atoms.get_volume()) * 4 * np.pi * dr * bin_edges[:-1] ** 2)
        coordination_numbers = np.sum(full_dist < cutoff, axis=1)
        return rdf, bin_edges, coordination_numbers

    def calculate_rdf(self, rmax:float, cutoff:float=2.0, dr:float=0.02):
        if isinstance(self.structure[0], ase.atom.Atom):
            rdf, bin_edges, coordination_numbers = self.calculate_single_rdf(self.structure.copy(), rmax, cutoff, dr)
        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            rdf = np.zeros(len(bins) - 1)
            for idx, atoms in enumerate(self.structure):
                if idx == 0: 
                    nions = atoms.get_global_number_of_atoms()
                    coordination_numbers = np.zeros(nimg*nions)
                single_rdf, bin_edges, single_coordination_numbers = self.calculate_single_rdf(atoms, rmax, cutoff, dr)
                rdf += single_rdf
                coordination_numbers[idx*nions: (idx+1)*nions] = single_coordination_numbers
            rdf /= nimg
        cn_distribution = np.histogram(coordination_numbers, bins=np.arange(self.cn_lim[0], self.cn_lim[1], 1))
        cn_sum = np.sum(cn_distribution[0])
        return np.column_stack((bin_edges[:-1], rdf)), np.column_stack((cn_distribution[1][:-1], cn_distribution[0], cn_distribution[0]/cn_sum))

if __name__ == "__main__":
    main()
