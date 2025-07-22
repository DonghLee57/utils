import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from collections import Counter
from typing import Type

def main():
    np.random.seed(412)
    structure_A = 'your_input_A'
    structure_B = 'your_input_B'
    
    atoms_A = read(structure_A, format='vasp')
    atoms_A = atoms_A.repeat([3, 3, 3])
    atoms_A.rattle(stdev=0.5)
    write('source_A.cif', atoms_A)

    #atoms_B = read(structure_B, format='vasp')
    atoms_B = Atoms('H', cell=[[14, 0, 0], [2, 7, 0], [1, 1, 10]], pbc=True)
    write('source_B.cif', atoms_B)

    try:
        result_slab = find_composition_matched_subcell(atoms_A, atoms_B, search_step=2.0, min_distance=0.7)
        
        write('result.cif', result_slab)
        print("\nSuccessfully created files: source_A.cif, result_slab_with_dist_check.cif")
        print(f"Original composition: {Counter(atoms_A.get_chemical_symbols())}")
        print(f"Resulting composition: {Counter(result_slab.get_chemical_symbols())}")

    except ValueError as e:
        print(e)

def find_composition_matched_subcell(
    source_atoms: Atoms,
    template_atoms: Atoms,
    search_step: float = 1.0,
    min_distance: float = 0.0
) -> Atoms:
    """
    Finds a subcell within a supercell of `source_atoms` that matches the
    lattice of `template_atoms` and the chemical composition of `source_atoms`.
    It also ensures that no two atoms in the final structure are closer than
    the specified `min_distance`.

    Parameters
    ----------
    source_atoms : ase.Atoms
        The source structure whose atoms and composition will be used.
    template_atoms : ase.Atoms
        A template structure whose cell (lattice) defines the shape to cut.
    search_step : float, optional
        The step size (in Å) for moving the template cell's origin during the search.
        A smaller value is more precise but slower. Defaults to 1.0.
    min_distance : float, optional
        The minimum allowed distance between any two atoms in the final slab.
        If set to 0.0, this check is skipped. Defaults to 0.0.

    Returns
    -------
    ase.Atoms
        A new Atoms object containing the atoms from the found region,
        with the cell set to the template's cell.

    Raises
    ------
    ValueError
        If no region satisfying both composition and distance criteria can be found.
    """

    source_symbols = source_atoms.get_chemical_symbols()
    if not source_symbols:
        raise ValueError("The `source_atoms` object contains no atoms.")

    target_counts = Counter(source_symbols)
    target_total_atoms = len(source_symbols)
    target_ratios = {el: count / target_total_atoms for el, count in target_counts.items()}

    source_cell = source_atoms.get_cell()
    template_cell = template_atoms.get_cell()

    inv_source_cell = np.linalg.inv(source_cell)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ]) @ template_cell
    frac_coords_vertices = vertices @ inv_source_cell
    
    span = np.max(frac_coords_vertices, axis=0) - np.min(frac_coords_vertices, axis=0)
    replication = np.ceil(span).astype(int) + 2
    
    print(f"Creating a supercell of size: {replication}")
    supercell = make_supercell(source_atoms, np.diag(replication))
    supercell_positions = supercell.get_positions()
    supercell_symbols = np.array(supercell.get_chemical_symbols())

    inv_template_cell = np.linalg.inv(template_cell)
    search_range = np.linalg.norm(supercell.get_cell(), axis=1) - np.linalg.norm(template_cell, axis=1)

    for ox in np.arange(0, max(0, search_range[0]), search_step):
        for oy in np.arange(0, max(0, search_range[1]), search_step):
            for oz in np.arange(0, max(0, search_range[2]), search_step):
                origin = np.array([ox, oy, oz])
                relative_pos = supercell_positions - origin
                frac_coords_in_template = relative_pos @ inv_template_cell
                
                mask = np.all((frac_coords_in_template >= 0) & (frac_coords_in_template < 1), axis=1)
                indices_inside = np.where(mask)[0]
                
                if len(indices_inside) > 0:
                    sub_symbols = supercell_symbols[indices_inside]
                    sub_counts = Counter(sub_symbols)
                    sub_total_atoms = len(sub_symbols)

                    if set(sub_counts.keys()) != set(target_ratios.keys()):
                        continue

                    is_match = all(
                        np.isclose(sub_counts[el] / sub_total_atoms, target_ratios[el], atol=1e-3)
                        for el in target_ratios
                    )
                    
                    if is_match:
                        candidate_atoms = supercell[indices_inside]
                        
                        if min_distance > 0.0:
                            if len(candidate_atoms) > 1:
                                distances = candidate_atoms.get_all_distances(mic=False)
                                np.fill_diagonal(distances, np.inf) # Ignore self-distance
                                if np.min(distances) < min_distance:
                                    continue # This candidate is invalid, try next origin
                        
                        print(f"Found a valid region at origin: {origin.round(2)}")
                        print(f"Number of atoms in slab: {sub_total_atoms}")
                        
                        found_atoms = candidate_atoms
                        found_atoms.set_cell(template_cell)
                        found_atoms.set_pbc(True)
                        found_atoms.wrap()
                        return found_atoms

    raise ValueError("Could not find a region satisfying both composition and minimum distance criteria.")

if __name__ == '__main__':
    main()
