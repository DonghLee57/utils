import numpy as np 
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import make_supercell
from ase.io import write
import spglib

def get_equivalent_planes(unit_cell, surface_energies):
    """결정 대칭성을 고려하여 등가면을 확장합니다."""
    lattice = unit_cell.get_cell()
    positions = unit_cell.get_scaled_positions()
    numbers = unit_cell.get_atomic_numbers()
    dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    rotations = dataset['rotations']
    
    expanded_energies = {}
    for hkl, energy in surface_energies.items():
        hkl_vec = np.array(hkl)
        unique_hkls = set()
        for rot in rotations:
            new_hkl = tuple(np.dot(hkl_vec, rot))
            unique_hkls.add(new_hkl)
        for sym_hkl in unique_hkls:
            if sym_hkl not in expanded_energies:
                expanded_energies[sym_hkl] = energy
    return expanded_energies

def create_terminated_wulff(unit_cell, surface_energies, scale, termination_element=None):
    """
    특정 원소 터미네이션을 고려한 범용 Wulff Construction 함수
    
    Args:
        unit_cell (ase.Atoms): 단위 셀
        surface_energies (dict): Miller index와 에너지
        scale (float): 클러스터 크기 조절 인자
        termination_element (str): 표면에 노출되길 원하는 원소 기호 (예: 'Ti')
    """
    full_surfaces = get_equivalent_planes(unit_cell, surface_energies)
    reciprocal_cell = unit_cell.get_reciprocal_cell()
    
    planes = []
    for hkl, energy in full_surfaces.items():
        normal = np.dot(hkl, reciprocal_cell)
        norm = np.linalg.norm(normal)
        if norm < 1e-8: continue
        unit_normal = normal / norm
        dist = scale * energy
        planes.append({'n': unit_normal, 'd': dist, 'hkl': hkl})

    max_dist = scale * max(full_surfaces.values())
    cell_lengths = np.linalg.norm(unit_cell.get_cell(), axis=1)
    n_reps = int(np.ceil(max_dist * 2.2 / min(cell_lengths)))
    supercell = make_supercell(unit_cell, np.eye(3) * (2 * n_reps + 1))
    supercell.center(about=(0, 0, 0))
    
    positions = supercell.get_positions()
    symbols = np.array(supercell.get_chemical_symbols())
    
    mask = np.ones(len(positions), dtype=bool)
    for p in planes:
        mask &= (np.dot(positions, p['n']) <= (p['d'] + 1e-6))
    
    if termination_element:
        # 각 평면의 경계(표면)에 있는 원자 중 지정된 원소가 아닌 것 제거
        tol = 0.5 # Angstrom
        to_remove = np.zeros(len(positions), dtype=bool)
        
        for p in planes:
            dist_from_plane = np.dot(positions, p['n'])
            on_surface = (dist_from_plane > (p['d'] - tol)) & (dist_from_plane <= (p['d'] + 1e-6))
            
            wrong_element = (symbols != termination_element)
            to_remove |= (on_surface & wrong_element & mask)
        
        mask &= ~to_remove

    cluster = supercell[mask]
    cluster.translate(-np.mean(cluster.get_positions(), axis=0))
    return cluster

def plot_wulff_polar(unit_cell, surface_energies, zone_axis=(0, 0, 1)):
    """
    특정 zone axis를 기준으로 표면 에너지의 Polar plot을 생성합니다.
    """
    full_surfaces = get_equivalent_planes(unit_cell, surface_energies)
    reciprocal_cell = unit_cell.get_reciprocal_cell()
    
    # 2. Zone axis를 Cartesian 좌표로 변환 및 투영 평면 설정
    # zone_axis 방향의 벡터와 수직인 평면 상의 벡터들을 추출
    z_vec = np.dot(zone_axis, reciprocal_cell)
    z_vec /= np.linalg.norm(z_vec)
    
    # 평면 구성을 위한 임의의 x, y축 생성 (Gram-Schmidt)
    x_vec = np.array([1, 0, 0]) if abs(z_vec[0]) < 0.9 else np.array([0, 1, 0])
    x_vec = np.cross(x_vec, z_vec)
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(z_vec, x_vec)

    angles = []
    energies = []
    labels = []

    for hkl, energy in full_surfaces.items():
        # 각 Miller 지수의 법선 벡터 계산
        n_vec = np.dot(hkl, reciprocal_cell)
        norm = np.linalg.norm(n_vec)
        if norm < 1e-8: continue
        n_unit = n_vec / norm
        
        # n_unit 벡터가 우리가 설정한 투영 평면(x_vec, y_vec)에 있는지 확인
        # (zone_axis와 수직인 면들만 필터링)
        projection_z = np.dot(n_unit, z_vec)
        
        if abs(projection_z) < 1e-3: # 평면에 평행한 법선 벡터인 경우
            proj_x = np.dot(n_unit, x_vec)
            proj_y = np.dot(n_unit, y_vec)
            
            angle = np.arctan2(proj_y, proj_x)
            angles.append(angle)
            energies.append(energy)
            labels.append(str(hkl))

    if not angles:
        print(f"해당 zone axis {zone_axis} 평면 상에 존재하는 Miller 지수가 없습니다.")
        return

    sorted_indices = np.argsort(angles)
    plot_angles = np.array(angles)[sorted_indices]
    plot_energies = np.array(energies)[sorted_indices]
    
    plot_angles = np.append(plot_angles, plot_angles[0] + 2*np.pi)
    plot_energies = np.append(plot_energies, plot_energies[0])

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    ax.plot(plot_angles, plot_energies, 'o-', linewidth=2, label='Surface Energy ($\gamma$)')
    ax.fill(plot_angles, plot_energies, alpha=0.1)
    
    for a, e, l in zip(angles, energies, labels):
        ax.annotate(l, (a, e), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_title(f"Surface Energy Polar Plot (Zone Axis: {zone_axis})", pad=20)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    from ase.build import bulk
    
    tin = bulk('TiN', 'rocksalt', a=4.24)
    
    energies = {
        (1, 1, 1): 0.8,
        (1, 0, 0): 1.0
    }
    plot_wulff_polar(tin, energies, zone_axis=(1, 1, 1))

    ti_terminated = create_terminated_wulff(tin, energies, scale=12.0, termination_element='Ti')
    write('Ti_terminated.vasp', ti_terminated, format='vasp')

    n_terminated = create_terminated_wulff(tin, energies, scale=12.0, termination_element='N')
    write('N_terminated.vasp', n_terminated, format='vasp')
    
    print(f"Ti-Terminated 원자 수: {len(ti_terminated)}")
    print(f"N-Terminated 원자 수: {len(n_terminated)}")
    
    from collections import Counter
    print("Ti-Terminated 조성:", Counter(ti_terminated.get_chemical_symbols()))
    print("N-Terminated 조성:", Counter(n_terminated.get_chemical_symbols()))
