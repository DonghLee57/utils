# python3.x
# - Generates polycrystal with multiple-phase crystal.

import sys, os
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.transform import Rotation as R

config = {'grain_radius': 10,
          'cutoff_overlap': 2.0,
          'pad_width': 0.5}
config['lattice'] = None #np.array([[20, 0, 0],
                    #          [0, 20, 0],
                    #          [0, 0, 20]])

def main()->None:
    """
    user_config: User-specified configurations. Optional.
    """
    if os.path.isfile('input.config'):
        user_config = 'input.config'
    else:
        user_config = None

    grain_config = get_grain_configuration(user_config)

    lat, seed_positions = estimate_cell_size_and_generate_seeds(grain_config)

    poly = Atoms(pbc=[True, True, True])
    poly.cell = lat.copy()
    vpts = create_periodic_voronoi(seed_positions, poly.cell)
    vor = Voronoi(vpts)
    
    # Load phases
    phases = load_phases(grain_config['structure_file'], poly.cell)
    IDX = 0
    for phase_index, phase in enumerate(phases):
        grain = phase.copy()
        com = grain.get_center_of_mass()
        fixed = np.where(grain_config.loc[phase_index, ['euler_angle_x', 'euler_angle_y', 'euler_angle_z']] != 'R')[0]
        rotation = R.random().as_euler('xyz', degrees=True)
        rotation[fixed] = grain_config.loc[phase_index, np.array(['euler_angle_x', 'euler_angle_y', 'euler_angle_z'])[fixed]]
        grain.euler_rotate(*rotation, center=com)
        grain.translate(vor.points[IDX] - com)
        ids = get_points_in_voronoi_cell(vor, IDX, grain.positions)
        grain = grain[ids]
        grain = remove_atoms_near_other_seeds(grain, seed_positions, IDX, config['pad_width'])
        poly += grain
        IDX += 1
    poly.wrap()
    dist = poly.get_all_distances(mic=True)
    id1 = (dist > 1E-3)*1
    id2 = (dist < config['cutoff_overlap'])
    ids = np.where( id1+id2==2 )[0]
    del poly[ids]
    poly = poly[poly.numbers.argsort()]
    write('output.vasp', images=poly, format='vasp')
    return None

#
def get_grain_configuration(user_config=None):
    """
    Generates a configuration dictionary with default settings.
    If user_config is provided, overrides the default settings with user-specified values.
    Parameters:
    - user_config: Dictionary containing user-specified configurations.
    Returns:
    - Dictionary with complete configuration for the polycrystalline structure generation.
    """
    # Default configuration
    cols = ['structure_file', 'number_of_structure', 'position_x', 'position_y', 'position_z', 'euler_angle_x', 'euler_angle_y', 'euler_angle_z']
    if user_config == None:
        tmp = pd.DataFrame(columns=cols)
        tmp.loc[0] = ['POSCAR', 4, 0, 0, 0, 0, 0, 0]
    else:
        tmp = pd.read_csv(user_config, delim_whitespace=True, names=cols)

    df = pd.DataFrame(columns=cols)
    row=0
    for i in range(len(tmp)):
        grains = tmp.loc[i,'number_of_structure']
        for j in range(grains):
            if grains == 1: 
                df.loc[row] = tmp.loc[i]
                df.loc[row,'number_of_structure'] = 1
            else:
                df.loc[row] = [tmp.loc[i,'structure_file'], 1, 'R', 'R', 'R', 'R', 'R', 'R']
            row += 1
    return df
    
def load_phases(phase_files, lattice_vectors):
    """
    Load phase structures from files and scale phases to match the longest lattice vector.
    
    Parameters:
    - phase_files: List of file paths for each phase's file.
    - lattice_vectors: Numpy array representing the lattice vectors of the target supercell.
    
    Returns:
    - phases: List of ASE Atoms objects for each phase, scaled to match the longest lattice vector.
    """
    longest_vector_length = np.max(np.linalg.norm(lattice_vectors, axis=1))
    phases = [read(phase_file) for phase_file in phase_files]
    for i, phase in enumerate(phases):
        current_longest = np.max(np.linalg.norm(phase.cell, axis=1))
        scaling_factor = np.ceil(longest_vector_length / current_longest).astype(int)
        scaling_matrix = np.eye(3) * scaling_factor
        phases[i] = make_supercell(phase, scaling_matrix)
    return phases

def estimate_cell_size_and_generate_seeds(grain_config):
    """
    Estimates the cell size needed to accommodate a specified number of grains with a given radius and generates seed positions.
    
    Parameters:
    - grain_config: DataFrame including information for grains.
    - grain_radius: The radius of each grain in angstrom.
    
    Returns:
    - A tuple containing the estimated cell size and an array of seed positions.
    """
    global config
    total_grains = grain_config['number_of_structure'].sum()
    volume_per_grain = 4 / 3 * np.pi * (config['grain_radius'] ** 3)
    total_volume = total_grains * volume_per_grain
    # Assuming a cubic cell
    cell_size = np.cbrt(total_volume)
    if type(config['lattice']) != type(None):
        lat = config['lattice']
    else:
        lat = np.eye(3) * cell_size
    seeds = np.empty((total_grains, 3))
    min_distance =  config['grain_radius']
    attempts = 0
    for i in range(total_grains):
        while True:
            new_seed = np.random.rand(3)
            fixed = np.where(grain_config.loc[i,['position_x','position_y','position_z']] != 'R')[0]
            new_seed[fixed] = list(map(float,grain_config.loc[i, np.array(['position_x','position_y','position_z'])[fixed]].values))
            new_seed = new_seed @ lat
            print(new_seed)
            if i == 0 or np.all(np.linalg.norm(seeds[:i] - new_seed, axis=1) >= min_distance):
                seeds[i] = new_seed
                break
            attempts += 1
            if attempts > 1000:
                raise ValueError("Too many attempts to place seeds.")
    return lat, seeds


def remove_atoms_near_other_seeds(grain, seed_positions, current_seed_index, pad_width):
    """
    Removes atoms from a grain if they are within the padding distance of any other seed position.
    
    Parameters:
    - grain: The ASE Atoms object representing the current grain.
    - seed_positions: Numpy array of all seed positions.
    - current_seed_index: The index of the current seed (grain) being processed.
    - pad_width: The padding width; atoms within this distance from any other seed will be removed.
    
    Returns:
    - The ASE Atoms object for the grain with atoms removed as necessary.
    """
    current_seed = seed_positions[current_seed_index]
    other_seeds = np.delete(seed_positions, current_seed_index, axis=0)
    tree = cKDTree(other_seeds)
    distances, _ = tree.query(grain.positions, k=1)
    atoms_to_keep = distances > pad_width
    return grain[atoms_to_keep]

def create_periodic_voronoi(points, lat_vec):
    """
    Generate a Voronoi diagram with periodic boundary conditions for a given set of points and lattice vectors.
    
    Parameters:
    - points: An array of Cartesian points within the original cell.
    - lat_vec: The lattice vectors of the cell as a 3x3 matrix.
    
    Returns:
    - A numpy array of extended points considering periodic boundary conditions.
    """
    extended_points = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue  # Skip the original cell
                displacement = np.matmul(np.array([dx, dy, dz]), lat_vec)
                extended_points.extend(points + displacement)
    extended_points = np.array(extended_points)
    return np.vstack((points, extended_points))

def get_points_in_voronoi_cell(vor, cell_index, points):
    """
    Identifies points that are within a specified Voronoi cell.
    
    Parameters:
    - vor: A Voronoi object from scipy.spatial.
    - cell_index: Index of the generating point for the target Voronoi cell.
    - points: An array of points to test, in Cartesian coordinates.
    
    Returns:
    - An index array of points within the specified Voronoi cell.
    """
    distances_squared = np.sum((vor.points[:, np.newaxis, :] - points) ** 2, axis=2)
    closest_indices = np.argmin(distances_squared, axis=0)
    return np.where(closest_indices == cell_index)[0]

#
if __name__ == "__main__":
    main()
