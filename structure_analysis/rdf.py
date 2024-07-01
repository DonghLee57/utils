import numpy as np
from ase.io import read, write
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main():
    my = StructureAnalysis('POSCAR', 'vasp')

    rdf, cn_distribution = my.calculate_rdf(6, 2.5)
    np.savetxt('rdf.out', rdf, fmt='%.4f')
    np.savetxt('cn_distribution.out', cn_distribution, fmt='%.4f')

    return 0

class StructureAnalysis:
    def __init__(self, filename, fileformat, index='-1'):
        self.structure = read(filename, index=index, format=fileformat)

    def calculate_rdf(self, rmax, cutoff, dr=0.02):
        bins = np.arange(dr / 2, rmax + dr / 2, dr)
        rdf = np.zeros(len(bins) - 1)

        if isinstance(self.structure[0], ase.atom.Atom):
            atoms = self.structure.copy()
            if rmax > atoms.get_cell().diagonal().min() / 2:
                print('WARNING: The input maximum radius is over the half the smallest cell dimension.')

            nions = atoms.get_global_number_of_atoms()
            # Split the atoms array into nearly equal parts for each MPI process
            local_nions = nions // size
            start = rank * local_nions
            end = nions if rank == size - 1 else (rank + 1) * local_nions
            local_dist = np.zeros((end - start, nions))
            local_coordination_numbers = np.zeros(end - start)
            for i in range(start, end):
                distances = atoms.get_distances(i, range(i, nions), mic=True)
                local_dist[i - start, i:nions] = distances
            full_dist = np.zeros((nions, nions))
            comm.Allgatherv(local_dist, full_dist)
            full_dist += full_dist.T - np.diag(np.diag(full_dist))
            np.fill_diagonal(full_dist, np.inf)

            res, bin_edges = np.histogram(full_dist, bins=bins)
            rdf += res / ((nions ** 2 / atoms.get_volume()) * 4 * np.pi * dr * bin_edges[:-1] ** 2)
            local_coordination_numbers = np.sum(local_dist < cutoff, axis=1)

        elif isinstance(self.structure[0], ase.atoms.Atoms):
            nimg = len(self.structure)
            for idx, atoms in enumerate(self.structure):
                if rmax > atoms.get_cell().diagonal().min() / 2:
                    print('WARNING: The input maximum radius is over the half the smallest cell dimension.')

                # Split the atoms array into nearly equal parts for each MPI process
                local_nions = nions // size
                start = rank * local_nions
                end = nions if rank == size - 1 else (rank + 1) * local_nions
                local_dist = np.zeros((end - start, nions))
                if idx == 0: local_coordination_numbers = np.zeros(end - start)
                for i in range(start, end):
                    distances = atoms.get_distances(i, range(i, nions), mic=True)
                    local_dist[i - start, i:nions] = distances
                full_dist = np.zeros((nions, nions))
                comm.Allgatherv(local_dist, full_dist)
                full_dist += full_dist.T - np.diag(np.diag(full_dist))
                np.fill_diagonal(full_dist, np.inf)
                res, bin_edges = np.histogram(full_dist, bins=bins)
                rdf += res / ((nions ** 2 / atoms.get_volume()) * 4 * np.pi * dr * bin_edges[:-1] ** 2)
                local_coordination_numbers = np.sum(local_dist < cutoff, axis=1)
            rdf /= nimg

        # Gather all coordination numbers at root process
        total_coordination_numbers = np.zeros(nions)
        comm.Gatherv(local_coordination_numbers, [total_coordination_numbers, local_nions, MPI.DOUBLE], root=0)

        # Only the root process should compute the final histogram
        if rank == 0:
            cn_distribution = np.histogram(total_coordination_numbers, bins='auto', density=True)
            return np.column_stack((bin_edges[:-1], rdf)), np.column_stack((cn_distribution[1][:-1], cn_distribution[0]))
        else:
            return None

if __name__ == "__main__":
    main()
