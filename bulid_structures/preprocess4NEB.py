import os
import numpy as np
from ase.io import read, write
from scipy.optimize import linear_sum_assignment

def main():
    poscar1 = 'POSCAR_i'
    poscar2 = 'POSCAR_f'
    z_threshold = 15.0
   
    a1, a2 = save_sorted(poscar1, poscar2, z_threshold)

    n_imgs = 12
    neb_images = interpolate_NEB_images(a1, a2, n_imgs)
    write('test.extxyz', neb_images, format='extxyz')

def sort_by_min_distance(ref_atoms, target_atoms, idx_ref, idx_target):
    """
    Sort specified atoms based on minimum distance criterion using Hungarian algorithm.
    
    Args:
        ref_atoms: Reference ASE atoms object
        target_atoms: Target ASE atoms object to be sorted
        idx_ref: List of indices for reference atoms
        idx_target: List of indices for target atoms to be sorted
    
    Returns:
        List of sorted indices from target_atoms
    """
    ref_pos = ref_atoms.get_positions()[idx_ref]
    target_pos = target_atoms.get_positions()[idx_target]
    ref_cell = ref_atoms.get_cell()
    target_cell = target_atoms.get_cell()
    
    # Check if unit cells are identical
    cell_tolerance = 1e-6
    if not np.allclose(ref_cell, target_cell, atol=cell_tolerance):
        print("WARNING: Unit cells of reference and target structures are different!")
        print("This may lead to incorrect periodic boundary condition corrections.")
        print("Please ensure both structures have identical unit cells for accurate sorting.")
        print(f"Reference cell:\n{ref_cell}")
        print(f"Target cell:\n{target_cell}")
    
    # Use reference cell for PBC correction (assuming they should be identical)
    cell = ref_cell
    
    # Calculate distance matrix with periodic boundary conditions
    dmat = np.zeros((len(ref_pos), len(target_pos)))
    for i, rp in enumerate(ref_pos):
        diff = target_pos - rp
        # Minimum image convention (PBC correction)
        diff -= np.round(diff @ np.linalg.inv(cell)) @ cell
        dmat[i, :] = np.linalg.norm(diff, axis=1)
    
    # Solve assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(dmat)
    
    # Return sorted indices in original target_atoms order
    return [idx_target[j] for j in col_ind]

def set_origin(atoms, index):
    """
    Translate the entire structure so that the specified atom is at the origin.
    
    Args:
        atoms: ASE atoms object
        index: Index of the atom to be placed at origin
    
    Returns:
        ASE atoms object with translated positions
    """
    return atoms.translate(-atoms.get_positions()[index])

def save_sorted(initial_path, final_path, z_threshold):
    """
    Sort atomic structures for NEB calculations.
    Separates fixed (slab) and mobile atoms based on z-coordinate threshold,
    then sorts fixed atoms to minimize displacement between initial and final states.
    """
    # File paths
    poscar1_path = initial_path        # Initial state POSCAR
    poscar2_path = final_path          # Final state POSCAR  
    output1_path = f"{initial_path}_sorted" # Output file name for initial
    output2_path = f"{final_path}_sorted"   # Output file name for final
   
    try:
        atoms1 = read(poscar1_path)
        atoms2 = read(poscar2_path)
        pos1 = atoms1.get_positions()
        pos2 = atoms2.get_positions()
        
        # Separate indices for atoms1 (initial structure)
        fixed_idx1 = [i for i, p in enumerate(pos1) if p[2] >= z_threshold]
        mobile_idx1 = [i for i, p in enumerate(pos1) if p[2] < z_threshold]
        
        # Separate indices for atoms2 (final structure)
        fixed_idx2 = [i for i, p in enumerate(pos2) if p[2] >= z_threshold]
        mobile_idx2 = [i for i, p in enumerate(pos2) if p[2] < z_threshold]
        
        # Check if atom counts match (crucial for consistency)
        assert len(fixed_idx1) == len(fixed_idx2), \
            f"Fixed layer atom count mismatch: initial={len(fixed_idx1)}, final={len(fixed_idx2)}"
        assert len(mobile_idx1) == len(mobile_idx2), \
            f"Mobile layer atom count mismatch: initial={len(mobile_idx1)}, final={len(mobile_idx2)}"
            
        print(f"Structure analysis:")
        print(f"  Fixed layer atoms: {len(fixed_idx1)} (z >= {z_threshold} Å)")
        print(f"  Mobile layer atoms: {len(mobile_idx1)} (z < {z_threshold} Å)")
        
        sorted_fixed_idx2 = sort_by_min_distance(atoms1, atoms2, fixed_idx1, fixed_idx2)
        final_order_idx2 = sorted_fixed_idx2 + mobile_idx2
        atoms2_sorted = atoms2[final_order_idx2]
        
        final_order_idx1 = fixed_idx1 + mobile_idx1
        atoms1_sorted = atoms1[final_order_idx1]
        
        # 6. Save both sorted structures to files
        write(output1_path, atoms1_sorted, format="vasp")
        write(output2_path, atoms2_sorted, format="vasp")
        
        print(f"Structure sorting completed:")
        print(f"  Initial structure saved as '{output1_path}'")
        print(f"  Final structure saved as '{output2_path}'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
    except AssertionError as e:
        print(f"Error: {e}")
        print("Please check your input structures and z_threshold value.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
    return atoms1_sorted, atoms2_sorted

def interpolate_NEB_images(atoms_initial, atoms_final, n_images):
    """
    ssNEB용 이미지 생성: 
    주기경계조건(PBC)과 cell 보간을 모두 고려해 원자의 최소 이동 경로로 중간 이미지 생성.
    
    Args:
        atoms_initial: ASE Atoms 객체 (초기 구조)
        atoms_final: ASE Atoms 객체 (최종 구조)
        n_images: 중간 이미지 개수 (initial/final 불포함)
    
    Returns:
        전체 NEB 경로 (initial + 중간 이미지들 + final) 리스트
    """
    images = [atoms_initial.copy()]
    pos_i = atoms_initial.get_positions()
    pos_f = atoms_final.get_positions()
    cell_i = atoms_initial.get_cell()
    cell_f = atoms_final.get_cell()
    pbc = atoms_initial.get_pbc()

    frac_i = np.linalg.solve(cell_i.T, pos_i.T).T
    frac_f = np.linalg.solve(cell_f.T, pos_f.T).T

    wrapped_frac_f = frac_f.copy()
    for i in range(len(frac_i)):
        dvec = frac_f[i] - frac_i[i]
        dvec -= np.round(dvec)
        wrapped_frac_f[i] = frac_i[i] + dvec

    for j in range(1, n_images + 1):
        cell_interp = cell_i + (cell_f - cell_i) * (j / (n_images + 1))
        frac_interp = frac_i + (wrapped_frac_f - frac_i) * (j / (n_images + 1))
        frac_interp = frac_interp % 1.0
        pos_interp = np.dot(frac_interp, cell_interp)
        img = atoms_initial.copy()
        img.set_cell(cell_interp)
        img.set_positions(pos_interp)
        img.set_pbc(pbc)
        images.append(img)
        if os.path.isdir(f"{j:02d}"):
            os.makedirs(f"{j:02d}",exist_ok=True)
            write(f'{j:02d}/POSCAR', img, format='vasp')
        
    images.append(atoms_final.copy())
    return images

if __name__ == '__main__':
    main()
