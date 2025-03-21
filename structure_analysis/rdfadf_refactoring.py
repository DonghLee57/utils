import sys
import numpy as np
import ase
from ase.io import read, write
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def main():
    my = StructureAnalysis()
    my.load_structure(sys.argv[1], sys.argv[2])
    #my.load_structure('POSCAR', 'vasp')
    #my.load_structure('input.lammps', 'lammps-data')
    
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
    def __init__(self):
        self.structure = None
        self.cn_lim = [0, 10]
        self.distance_matrix = None

    def load_structure(self, filename: str, file_format: str, **kwargs):
        """
        Load atomic structure from file.
        
        Args:
            filename (str): Path to the input file.
            file_format (str): Format of the input file ('vasp' or 'lammps-data').
            **kwargs: Additional keyword arguments for ase.io.read function.
        """
        # Reinitialize the distance_matrix each time the structure is reloaded
        self.distance_matrix = None  
        
        if file_format not in ['vasp', 'lammps-data','extxyz']:
            raise ValueError("Unsupported file format. Use 'vasp', 'lammps-data', or 'extxyz'.")
        default_args = {
            'vasp': {'index': None},
            'lammps-data': {'index': None,
                            'style': 'atomic'},
            'extxyz': {'index': None},
        }

        # Merge default arguments with user-provided kwargs
        read_args = {**default_args[file_format], **kwargs}
        try:
            self.structure = read(filename, format=file_format, **read_args)
        except Exception as e:
            raise IOError(f"Failed to load structure: {str(e)}")

    def calculate_distance_matrix(self, atoms: ase.atom.Atom):
        nions = atoms.get_global_number_of_atoms()
        local_nions = nions // size
        start = rank * local_nions
        end = nions if rank == size - 1 else (rank + 1) * local_nions
        local_dist = np.zeros((end - start, nions))
        for i in range(start, end):
            local_dist[i - start, i:nions] = atoms.get_distances(i, range(i, nions), mic=True)
        local_size = local_dist.size
        sizes = comm.allgather(local_size)
        global_size = sum(sizes)
        displacements = [sum(sizes[:i]) for i in range(size)]
        global_dist = np.zeros(global_size).reshape([-1, local_dist.shape[1]])
        comm.Allgatherv(sendbuf=local_dist, recvbuf=(global_dist, sizes, displacements, MPI.DOUBLE))
        global_dist += global_dist.T - np.diag(np.diag(global_dist))
        np.fill_diagonal(global_dist, np.inf)
        self.distance_matrix = global_dist
        return self.distance_matrix
    
    def calculate_single_rdf(self, atoms:ase.atom.Atom, rmax:float, cutoff:float, dr:float):
        if self.distance_matrix is None:
            self.distance_matrix = self.calculate_distance_matrix(atoms)
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        rdf = np.zeros(len(bins) - 1)
        if rmax > atoms.get_cell().diagonal().min() / 2:
            print('WARNING: The input maximum radius is over the half the smallest cell dimension.')
        global_dist = self.distance_matrix
        nions = atoms.get_global_number_of_atoms()
        res, bin_edges = np.histogram(global_dist, bins=bins)
        rdf += res / ((nions ** 2 / atoms.get_volume()) * 4 * np.pi * dr * bin_edges[:-1] ** 2)
        coordination_numbers = np.sum(global_dist < cutoff, axis=1)
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
        if self.distance_matrix is None:
            self.distance_matrix = self.calculate_distance_matrix(atoms)
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
        global_dist = self.distance_matrix[idA][:, idB]
        res, bin_edges = np.histogram(global_dist, bins=bins)
        prdf += res / (nelemA * nelemB / atoms.get_volume() * 4 * np.pi * dr * bin_edges[:-1] ** 2)
        if elemA == elemB:
            coordination_numbers = np.sum(global_dist < cutoff, axis=1) - 1
        else:
            coordination_numbers = np.sum(global_dist < cutoff, axis=1)
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
        if self.distance_matrix is None:
            self.distance_matrix = self.calculate_distance_matrix(atoms)
        theta = []
        symbol_idx = {s: [] for s in triplet}
        for idx, atom in enumerate(atoms):
            if atom.symbol in symbol_idx:
                symbol_idx[atom.symbol].append(idx)
        psize = len(symbol_idx[triplet[0]]) // size
        for c in range(rank * psize, (rank + 1) * psize if rank != size - 1 else len(symbol_idx[triplet[0]])):
            center_idx = symbol_idx[triplet[0]][c]
            distances = self.distance_matrix[center_idx]
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
