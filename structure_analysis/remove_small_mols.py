from mpi4py import MPI
import sys
import numpy as np
from ase.io import read, write
from ase.data import atomic_numbers, chemical_symbols
import torch
from torch_geometric.data import Data
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Pre-defined parameters
MINSIZE = 10    # Minimum size of a connected component to be retained in the graph
CUTOFF = 2.4    # Maximum distance between atoms to be considered connected in the graph network
PAIRCUTOFFS = {
    tuple(sorted(('Si','Si'))):2.4,
    tuple(sorted(('Si','H'))): 2.0}
OUTPUT = 'output'
# LAMMPS output setting - With N-element system, set [1:N+1]
SPECORDER = chemical_symbols[1:6] 

class UnionFind:
    def __init__(self, num_elements):
        self.root = np.arange(num_elements, dtype=int)
        self.rank = np.ones(num_elements, dtype=int)

    def find(self, x):
        if self.root[x] != x:
            self.root[x] = self.find(self.root[x])
        return self.root[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1

def process_graph(edges, num_nodes):
    uf = UnionFind(num_nodes)
    for x, y in edges:
        uf.union(x, y)
    return uf

def create_graph_object(structure_file, pair_cutoffs):
    atoms = read(structure_file)
    #atoms = read(structure_file, style='atomic', format='lammps-data')
    num_atoms = len(atoms)

    atoms_per_proc = num_atoms // size
    start = rank * atoms_per_proc
    end = (rank + 1) * atoms_per_proc if rank != size - 1 else num_atoms

    local_indices = []
    for i in range(start, end):
        local_distances = atoms.get_distances(i, range(num_atoms), mic=True)
        for j in range(num_atoms):
            if i != j:
                pairs = tuple(sorted((atoms[i].symbol, atoms[j].symbol)))
                cutoff = pair_cutoffs.get(pairs, CUTOFF)
                if local_distances[j] <= cutoff:
                    local_indices.append((i, j))

    all_indices = comm.gather(local_indices, root=0)
    if rank == 0:
        edge_indices = [pair for sublist in all_indices for pair in sublist]
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(atoms.numbers, dtype=torch.float)
        graph = Data(x=node_features, edge_index=edge_index_tensor)
        return graph, atoms
    return None, None

def main():
    structure_file = sys.argv[1]
    graph, atoms = create_graph_object(structure_file, PAIRCUTOFFS)
    if rank == 0:
        num_nodes = graph.num_nodes
        edges = graph.edge_index.t().tolist()
    else:
        num_nodes = None
        edges = None
    num_nodes = comm.bcast(num_nodes, root=0)
    edges = comm.bcast(edges, root=0)

    local_uf = process_graph(edges, num_nodes)

    global_roots = np.empty(num_nodes, dtype=int)
    comm.Allreduce(local_uf.root, global_roots, op=MPI.MAX)

    if rank == 0:
        final_uf = UnionFind(num_nodes)
        final_uf.root = global_roots
        unique_components = set(final_uf.find(x) for x in range(num_nodes))
        component_dict = {root: [] for root in unique_components}

        for node in range(num_nodes):
            root = final_uf.find(node)
            component_dict[root].append(node)

        print("Number of connected components:", len(unique_components))

        nodes_to_remove = []
        for root, component in component_dict.items():
            component_size = len(component)
            print("Component size:", component_size)
            if component_size <= MINSIZE:
                print("Small component atoms:", atoms[np.array(component)].symbols)
                nodes_to_remove.extend(component)

        if nodes_to_remove:
            atoms = atoms[[i for i in range(len(atoms)) if i not in nodes_to_remove]]

        write(OUTPUT+'.vasp', images=atoms, format='vasp', parallel=False)
        #write(OUTPUT+'.lammps', images=atoms, format='lammps-data', parallel=False, specorder=SPECORDER)
        #write(OUTPUT+'.extxyz', images=atoms, format='extxyz', parallel=False)
    return 0

if __name__ == "__main__":
    main()
