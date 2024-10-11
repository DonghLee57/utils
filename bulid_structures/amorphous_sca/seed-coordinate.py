import yaml
import re
import numpy as np
from ase import Atoms
from ase.io import write
from ase.data import atomic_masses, chemical_symbols
from ase.neighborlist import NeighborList
import scipy.constants as CONST

# Constants
MAX_ITERATIONS = 1000
ANGSTROM_TO_CM = 1E-8

def main():
    # Usage example
    input_data = load_input('input1.yaml')
    simulation = SCBuilder(input_data)
    simulation.run_simulation()
    atoms = simulation.get_structure()
    print(atoms)
    print(calculate_mass_density(atoms))
    write('test1.vasp', atoms, format='vasp')

    input_data = load_input('input2.yaml')
    simulation = SCBuilder(input_data)
    simulation.run_simulation()
    atoms = simulation.get_structure()
    print(atoms)
    print(calculate_mass_density(atoms))
    write('test2.vasp', atoms, format='vasp')

#
def load_input(filename: str) -> dict:
    """
    Load and process input data from a YAML file.
    """
    with open(filename, 'r') as file:
        input_data = yaml.safe_load(file)
    
    if 'lattice' in input_data and 'density' in input_data and 'chemical_formula' in input_data:
        # Case 1: Calculate num_atoms based on lattice, density, and chemical_formula
        input_data['lattice'] = process_lattice(input_data['lattice'])
        chemical_formula = parse_chemical_formula(input_data['chemical_formula'])
        input_data['atom_types'] = list(chemical_formula.keys())
        input_data['num_atoms'] = calculate_num_atoms(input_data['lattice'],
                                                      input_data['density'],
                                                      chemical_formula)
    elif 'density' in input_data and 'num_atoms' in input_data and 'atom_types' in input_data:
        # Case 2: Calculate lattice based on density, num_atoms, and atom_types
        input_data['lattice'] = calculate_lattice(input_data['num_atoms'], input_data['density'], input_data['atom_types'])
    else:
        raise ValueError("Input must contain either 'lattice', 'density', and 'chemical_formula' or 'density', 'num_atoms', and 'atom_types'")
    
    return input_data

def process_lattice(lattice):
    """
    Process the lattice parameter from input.
    Args:
        lattice: The lattice parameter. Can be a float or a 3x3 list/array.
    Returns:
        numpy.ndarray: A 3x3 array representing the cell.
    Raises:
        ValueError: If lattice is not in the correct format.
        TypeError: If lattice is not a float, list, or numpy array.
    """
    if isinstance(lattice, (float, int)):
        return np.eye(3) * float(lattice)
    elif isinstance(lattice, (list, np.ndarray)):
        lattice_array = np.array(lattice)
        if lattice_array.shape == (3, 3):
            return lattice_array
        else:
            raise ValueError("Lattice must be a 3x3 array if provided as a list or array")
    else:
        raise TypeError("Lattice must be a float, list, or numpy array")

def parse_chemical_formula(formula: str) -> dict:
    """
    Parse a chemical formula string into a dictionary.
    Args:
        formula (str): Chemical formula string.
    Returns:
        dict: Dictionary with elements as keys and their counts as values.
    """
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)
    return {element: int(count) if count else 1 for element, count in matches}

def calculate_num_atoms(lattice: np.ndarray, density: float, chemical_formula: dict) -> list:
    """
    Calculate the number of atoms for each element based on lattice, density, and chemical formula.
    """
    volume = np.abs(np.linalg.det(lattice))
    total_mass = sum(atomic_masses[chemical_symbols.index(elem)] * count for elem, count in chemical_formula.items())
    target_formula_units = (density * volume * (ANGSTROM_TO_CM**3) * CONST.N_A) / total_mass
    formula_units = max(1, round(target_formula_units))
    return [count * formula_units for count in chemical_formula.values()]

def calculate_lattice(num_atoms: list, density: float, atom_types: list) -> np.ndarray:
    """
    Calculate the lattice parameter based on density, num_atoms, and atom_types.
    Args:
        num_atoms (list): Number of atoms for each atom type.
        density (float): Target density in g/cm^3.
        atom_types (list): List of atom types.
    Returns:
        np.ndarray: A 3x3 array representing the cubic cell.
    """
    total_mass = sum(atomic_masses[chemical_symbols.index(atom)] * num for atom, num in zip(atom_types, num_atoms))
    total_atoms = sum(num_atoms)
    volume_cm3 = (total_mass / CONST.N_A) / density
    volume_angstrom3 = volume_cm3 / ANGSTROM_TO_CM**3
    lattice_parameter = np.cbrt(volume_angstrom3)
    return np.eye(3) * lattice_parameter

def calculate_mass_density(atoms: Atoms) -> float:
    """
    Calculate the mass density of an ASE Atoms object.
    Args:
        atoms (ase.Atoms): The Atoms object to calculate density for.
    Returns:
        float: The mass density in g/cm3.
    """
    total_mass = sum(atomic_masses[atoms.numbers]) / CONST.N_A
    volume = atoms.get_volume() * (ANGSTROM_TO_CM**3)
    return total_mass / volume  # g/cm3

class SCBuilder:
    def __init__(self, input_data: dict):
        self.lattice = input_data['lattice']
        self.atom_types = input_data['atom_types']
        self.num_atoms = input_data['num_atoms']
        self.d_min = input_data['d_min']
        self.d_max = input_data['d_max']
        self.prob_types = input_data['prob_types']
        self.prob_cn = input_data['prob_cn']
        self.atoms = Atoms(cell=self.lattice, pbc=True)
        self.cn_targets = []
        self.atom_counts = [0] * len(self.atom_types)
        print(self.num_atoms)
        self.total_atoms = sum(self.num_atoms)
        self.log = input_data.get('log', 0)  # Default is 0, set to 1 to enable logging
        if self.log:
            self.log_file = open('sca.log', 'w')

    def __del__(self):
        if self.log:
            self.log_file.close()

    def write_log(self, atom_index, atom_type, process_step, position):
        if self.log:
            log_entry = f"{atom_index} {atom_type} {process_step} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}\n"
            self.log_file.write(log_entry)
            self.log_file.flush()  # Ensure the log is written immediately

    def run_simulation(self):
        """Run the SCA simulation to generate the atomic structure."""
        atom_index = 0
        placed_atoms = 0
        while any(count < target for count, target in zip(self.atom_counts, self.num_atoms)):
            if atom_index == placed_atoms:
                if self.seed_step(atom_index):
                    placed_atoms += 1
            else:
                if self.coordinate_step(atom_index):
                    placed_atoms += 1
            atom_index += 1

    def seed_step(self, index: int) -> bool:
        """Perform a seed step in the SCA algorithm."""
        available_types = [i for i, (count, target) in enumerate(zip(self.atom_counts, self.num_atoms)) if count < target]
        atom_type_index = np.random.choice(available_types)
        atom_type = self.atom_types[atom_type_index]
        
        for _ in range(MAX_ITERATIONS):
            position = np.random.rand(3) @ self.lattice
            if self.check_seed_position(position, atom_type):
                self.atoms.append(atom_type)
                self.atoms.positions[-1] = position
                self.atom_counts[atom_type_index] += 1
                self.cn_targets.append(self.get_target_cn(atom_type))
                self.write_log(index, atom_type, "seed", position)
                return True
        
        print(f"Failed to place seed atom of type {atom_type} after {MAX_ITERATIONS} attempts at {index}")
        return False

    def check_seed_position(self, position: np.ndarray, atom_type: str) -> bool:
        """
        Check if a seed position is valid.

        Args:
            position (np.ndarray): Proposed position for the new atom.
            atom_type (str): Type of the atom to be placed.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        if len(self.atoms) == 0:
            return True

        for i, other_type in enumerate(self.atoms.get_chemical_symbols()):
            distance = self.pbc_distance(position, self.atoms.positions[i])
            min_dist = self.d_min[f'{atom_type}-{other_type}']
            if distance < min_dist:
                return False
        return True

    def pbc_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        diff = pos1 - pos2
        diff -= np.round(diff @ np.linalg.inv(self.lattice)) @ self.lattice
        return np.linalg.norm(diff)

    def coordinate_step(self, index: int) -> bool:
        """Perform a coordinate step in the SCA algorithm."""
        if self.get_current_cn(index) >= self.cn_targets[index]:
            return False

        atom_type = self.atom_types[self.atoms[index].symbol]
        neighbor_type = self.choose_neighbor_type(atom_type)

        for _ in range(MAX_ITERATIONS):
            if sum(self.atom_counts) >= sum(self.num_atoms):
                return False

            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            distance = np.random.uniform(
                self.d_min[f'{atom_type}-{neighbor_type}'],
                self.d_max[f'{atom_type}-{neighbor_type}']
            )

            new_position = self.atoms.positions[index] + direction * distance
            new_position = np.dot(new_position, np.linalg.inv(self.lattice)) % 1
            new_position = np.dot(new_position, self.lattice)

            if self.check_new_position(new_position, neighbor_type):
                self.atoms.append(neighbor_type)
                self.atoms.positions[-1] = new_position
                self.atom_counts[self.atom_types.index(neighbor_type)] += 1
                self.cn_targets.append(self.get_target_cn(neighbor_type))
                if self.log:
                    self.write_log(len(self.atoms)-1, neighbor_type, "coordinate", new_position)
                return True

        return False

    def get_target_cn(self, atom_type: str) -> int:
        """
        Get the target coordination number for an atom type.

        Args:
            atom_type (str): Type of the atom.

        Returns:
            int: Target coordination number.
        """
        cn_values = list(self.prob_cn[atom_type].keys())
        cn_probabilities = list(self.prob_cn[atom_type].values())
        return np.random.choice(cn_values, p=cn_probabilities)

    def choose_neighbor_type(self, atom_type: str) -> str:
        """
        Choose a neighbor type based on probabilities.

        Args:
            atom_type (str): Type of the central atom.

        Returns:
            str: Chosen neighbor atom type.
        """
        probs = [self.prob_types[atom_type][t] for t in self.atom_types]
        return np.random.choice(self.atom_types, p=probs)

    def check_distances(self, atom_index: int) -> bool:
        """
        Check if an atom's distances to all other atoms are within allowed ranges.

        Args:
            atom_index (int): Index of the atom to check.

        Returns:
            bool: True if all distances are valid, False otherwise.
        """
        atom_type = self.atoms[atom_index].symbol

        for i, other_type in enumerate(self.atoms.get_chemical_symbols()):
            if i != atom_index:
                distance = self.atoms.get_distance(i, atom_index, mic=True)
                min_dist = self.d_min[f'{atom_type}-{other_type}']
                max_dist = self.d_max[f'{atom_type}-{other_type}']

                if distance < min_dist or distance > max_dist:
                    return False

        return True

    def get_current_cn(self, atom_index: int) -> int:
        """
        Get the current coordination number of an atom.

        Args:
            atom_index (int): Index of the atom.

        Returns:
            int: Current coordination number.
        """
        cutoff = max(max(self.d_max.values()), max(self.d_min.values()))
        nl = NeighborList([cutoff/2]*len(self.atoms), self_interaction=False, bothways=True)
        nl.update(self.atoms)
        indices, offsets = nl.get_neighbors(atom_index)
        return len(indices)

    def get_structure(self) -> Atoms:
        """
        Get the final atomic structure.

        Returns:
            Atoms: ASE Atoms object representing the final structure.
        """
        self.atoms = self.atoms[self.atoms.numbers.argsort()]
        return self.atoms

#
if __name__ == "__main__":
    main()
