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

    prdf, cn_distribution = my.calculate_prdf(('Si','Si'),6, 2.5, 0.005)
    np.savetxt('prdf.out', prdf, fmt='%.4f')
    np.savetxt('cn_distribution.out', cn_distribution, fmt='%.4f')
    
    adf = my.calculate_adf(triplet=('Si', 'Si', 'Si'), cutoff=[2.5, 3.0])
    np.savetxt('adf.out', adf, fmt='%.4f')
    
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

    def calculate_rdf(self, rmax, cutoff=2.0, dr=0.02):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        if isinstance(self.structure[0], ase.atom.Atom):
            rdf, bin_edges, coordination_numbers = self.calculate_single_rdf(self.structure, rmax, cutoff, dr)
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

    def calculate_single_prdf(self, atoms:ase.atom.Atom, targets:tuple, rmax:float, cutoff:float, dr:float):
        (elemA, elemB) = targets
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        prdf = np.zeros(len(bins) - 1)
        if rmax > atoms.get_cell().diagonal().min() / 2:
            print('WARNING: The input maximum radius is over the half the smallest cell dimension.')
        sym = np.array(atoms.get_chemical_symbols())
        idA = np.where( sym == elemA )[0]
        nelemA = len(idA)
        idB = np.where( sym == elemB )[0]
        nelemB = len(idB)
        full_dist = np.zeros((nelemA, nelemB))
        local_nions = nelemA // size
        start = rank * local_nions
        end = nions if rank == size - 1 else (rank + 1) * local_nions
        local_dist = np.zeros((end - start, nelemB))
        for i in range(start, end):
            distances = atoms.get_distances(i, idB, mic=True)
            local_dist[i - start] = distances
        comm.Allgatherv(local_dist, full_dist)
        res, bin_edges = np.histogram(full_dist, bins=bins)
        prdf += res / (nelemA * nelemB / atoms.get_volume() * 4 * np.pi * dr * bin_edges[:-1]**2)
        if elemA == elemB:
            coordination_numbers = np.sum(full_dist < cutoff, axis=1) - 1
        else:
            coordination_numbers = np.sum(full_dist < cutoff, axis=1)
        return prdf, bin_edges, coordination_numbers

    def calculate_prdf(self, targets:tuple, rmax:float, cutoff:float=2.0, dr:float=0.02):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        if isinstance(self.structure[0], ase.atom.Atom):
            prdf, bin_edges, coordination_numbers = self.calculate_single_prdf(self.structure, targets, rmax, cutoff, dr)
        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            prdf = np.zeros(len(bins) - 1)
            for idx, atoms in enumerate(self.structure):
                if idx == 0: 
                    nions = atoms.get_global_number_of_atoms()
                    coordination_numbers = np.zeros(nimg*nions)
                single_prdf, bin_edges, single_coordination_numbers = self.calculate_single_prdf(atoms, targets, rmax, cutoff, dr)
                prdf += single_prdf
                coordination_numbers[idx*nions: (idx+1)*nions] = single_coordination_numbers
            prdf /= nimg
        cn_distribution = np.histogram(coordination_numbers, bins=np.arange(self.cn_lim[0], self.cn_lim[1], 1))
        cn_sum = np.sum(cn_distribution[0])
        return np.column_stack((bin_edges[:-1], prdf)), np.column_stack((cn_distribution[1][:-1], cn_distribution[0], cn_distribution[0]/cn_sum))    
    
    def calculate_angles(self, atoms, triplet, cutoff):
        theta = []
        symbol_idx = {s: [] for s in triplet}
        for idx, atom in enumerate(atoms):
            if atom.symbol in symbol_idx:
                symbol_idx[atom.symbol].append(idx)

        psize = len(symbol_idx[triplet[0]]) // size
        for c in range(rank * psize, (rank + 1) * psize if rank != size - 1 else len(symbol_idx[triplet[0]])):
            center_idx = symbol_idx[triplet[0]][c]
            distances = atoms.get_distances(center_idx, range(len(atoms)), mic=True)
            vectors = atoms.get_distances(center_idx, range(len(atoms)), mic=True, vector=True)
            for n in symbol_idx[triplet[1]]:
                if n == c: continue
                vec1 = vectors[n]
                dist1 = distances[n]
                if dist1 < cutoff[0]:
                    for m in symbol_idx[triplet[2]]:
                        if m == c or m == n: continue
                        vec2 = vectors[m]
                        dist2 = distances[m]
                        if dist2 < cutoff[1]:
                            angle = np.arccos(np.clip(np.dot(vec1, vec2.T) / (dist1 * dist2), -1.0, 1.0))
                            theta.append(angle)
        return theta
        
    def calculate_adf(self, triplet, cutoff, angle_bins=np.arange(0,180.1,2), expr='degree'):
        if isinstance(self.structure[0], ase.atom.Atom):
            theta = self.calculate_angles(self.structure, triplet, cutoff)
        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            theta = []
            for idx, atoms in enumerate(self.structure):
                single_theta = self.calculate_angles(atoms, triplet, cutoff)
                single_theta = comm.reduce(single_theta, op=MPI.SUM, root=0)
                if rank == 0: theta += single_theta
        theta = comm.bcast(theta,root=0)
        if expr == 'degree':
            theta = np.degrees(theta)
        res, bin_edges = np.histogram(theta, bins=angle_bins, density=True)
        return np.column_stack((bin_edges[:-1], res))

if __name__ == "__main__":
    main()
