#!/usr/bin/env python3
"""Generate random amorphous structure with specified density and composition.

This script creates a random atomic configuration based on given element types,
quantities, and target density. Outputs are written in VASP POSCAR and LAMMPS formats.
"""

import glob
import os
import sys
from typing import List, Optional

import numpy as np
import scipy.constants as CONST
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_masses, atomic_names, atomic_numbers
from ase.io import read, write

AVOGADRO_NUMBER = 6.022e23  # Avogadro's number (mol-1)
ANGSTROM_TO_CM = 1e-8  # Conversion factor: Å → cm


def main() -> None:
    """Process command-line arguments and generate random atomic structure.

    Expects input format: <element1> <element2> ... <n1> <n2> ... <density>
    Example: Si O 2 4 2.3
    """
    r_crt = 1.7  # Critical radius (Å) for minimum interatomic distance
    inputs = sys.argv[1:]
    rho = float(inputs[-1])
    split_size = int((len(inputs) - 1) / 2)
    elements = inputs[:split_size]
    counts = list(map(int, inputs[split_size:-1]))

    # Sort elements by atomic number for consistent ordering
    atomic_nums = [atomic_numbers[e] for e in elements]
    sorted_indices = np.argsort(atomic_nums)
    sorted_elements = [elements[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # Create symbol list (e.g., ['Si', 'Si', 'O', 'O', 'O', 'O'])
    symbols = []
    for elem, count in zip(sorted_elements, sorted_counts):
        symbols.extend([elem] * count)

    # Generate and sort structure
    structure = random_poscar(rho, symbols, r_crt)
    structure = structure[structure.numbers.argsort()]

    print(f"Random structure generation - {calculate_mass_density(structure):.2f} g/cm3")
    write("POSCAR_generated.vasp", images=structure, format="vasp")
    write("bulk.lammps", images=structure, format="lammps-data", specorder=sorted_elements)


def get_density(atom_obj: Atoms) -> float:
    """Calculate mass density of an ASE Atoms object (redundant with calculate_mass_density).

    Args:
        atom_obj: Atomic structure object.

    Returns:
        Density in g/cm3.
    """
    total_mass = 0.0
    for symbol in atom_obj.get_chemical_symbols():
        total_mass += atomic_masses[atomic_numbers[symbol]]
    volume = atom_obj.get_volume()  # Å3
    return total_mass / volume / AVOGADRO_NUMBER * 1e24


def random_poscar(
    rho: float, symbols: List[str], r_crt: float, lat: Optional[np.ndarray] = None
) -> Atoms:
    """Generate random atomic configuration with minimum distance constraint.

    Args:
        rho: Target density (g/cm³).
        symbols: List of atomic symbols for all atoms.
        r_crt: Minimum allowed interatomic distance (Å).
        lat: Optional 3x3 lattice matrix. If None, creates cubic lattice.

    Returns:
        Randomly generated atomic structure.
    """
    max_iterations = 1000
    total_mass = 0.0
    for symbol in symbols:
        total_mass += atomic_masses[atomic_numbers[symbol]]

    # Create initial lattice
    if lat is not None and not np.all(lat == None):
        lattice = lat
    else:
        lat_val = (total_mass / rho / AVOGADRO_NUMBER * 1e24) ** (1.0 / 3.0)
        lattice = np.eye(3) * lat_val

    gen_poscar = None
    fixed_indices = []
    for i, symbol in enumerate(symbols):
        trials = 0
        if gen_poscar is None:
            gen_poscar = Atoms([symbol], positions=[[0, 0, 0]], pbc=True)
            gen_poscar.cell = lattice
            fixed_indices = [0]  # Fix first atom position
        else:
            while trials < max_iterations:
                trials += 1
                gen_poscar.append(symbol)
                gen_poscar.positions[-1] = np.matmul(np.random.random(3), gen_poscar.cell)
                min_dist = gen_poscar.get_distances(
                    -1, indices=list(range(len(gen_poscar) - 1)), mic=True
                ).min()
                if min_dist < r_crt:
                    del gen_poscar[-1]
                else:
                    break

    gen_poscar.set_constraint(FixAtoms(mask=[i in fixed_indices for i in range(len(gen_poscar))]))
    return gen_poscar


def calculate_mass_density(atoms: Atoms) -> float:
    """Calculate the mass density of an ASE Atoms object.

    Args:
        atoms: Atomic structure object.

    Returns:
        Mass density in g/cm3.
    """
    total_mass = sum(atomic_masses[atoms.numbers]) / CONST.N_A  # grams
    volume = atoms.get_volume() * (ANGSTROM_TO_CM**3)  # cm3
    return total_mass / volume


if __name__ == "__main__":
    main()
