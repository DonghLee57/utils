import numpy as np 
import matplotlib.pyplot as plt

class VaspParser:
    """
    POSCAR와 DOSCAR를 읽어 필요한 데이터를 추출하는 클래스입니다.
    """
    def __init__(self, poscar_path='POSCAR', doscar_path='DOSCAR', outcar_path='OUTCAR'):
        self.poscar_path = poscar_path
        self.doscar_path = doscar_path
        self.outcar_path = outcar_path
        self.z_coords = []
        self.c_lattice = 0.0
        self.energies = []
        self.site_dos = [] # [atom_index][energy_index]
        self.efermi = 0.0

    def read_poscar(self):
        """POSCAR에서 z축 격자 크기와 원자들의 z좌표를 읽습니다."""
        try:
            with open(self.poscar_path, 'r') as f:
                lines = f.readlines()
                
            scale = float(lines[1].strip())
            # 3번째 격자 벡터(c축)의 z성분 (대략적인 c축 길이)
            # 사방정계(Orthorhombic) 등을 가정. 기울어진 셀의 경우 벡터 연산 필요.
            c_vec = np.array([float(x) for x in lines[4].split()])
            self.c_lattice = np.linalg.norm(c_vec) * scale

            # 원자 개수 파악 (line 6 or 7 depending on VASP version)
            # VASP 5.x 이상에서는 원소 기호가 먼저 나옴
            if lines[5].strip().replace('.', '').isdigit(): # VASP 4 style
                natoms_list = [int(x) for x in lines[5].split()]
                start_line = 7 # direct/cartesian line
            else: # VASP 5 style
                natoms_list = [int(x) for x in lines[6].split()]
                start_line = 8
            
            total_atoms = sum(natoms_list)
            
            coord_type = lines[start_line-1].strip().lower()
            
            coords = []
            for i in range(total_atoms):
                line_idx = start_line + i
                vec = np.array([float(x) for x in lines[line_idx].split()[:3]])
                coords.append(vec)
            
            coords = np.array(coords)
            
            if 'd' in coord_type or 'f' in coord_type: # Direct or Fractional
                self.z_coords = coords[:, 2] * self.c_lattice
            else: # Cartesian
                self.z_coords = coords[:, 2]
                
            print(f"[INFO] POSCAR Loaded: {total_atoms} atoms, c-axis = {self.c_lattice:.4f} A")
            
        except Exception as e:
            print(f"[ERROR] Failed to read POSCAR: {e}")

    def read_doscar(self):
        """DOSCAR에서 에너지와 각 원자별 DOS를 읽습니다."""
        try:
            with open(self.doscar_path, 'r') as f:
                lines = f.readlines()
            
            # DOSCAR Header 정보 읽기 (6번째 줄)
            # [EMAX, EMIN, NEDOS, EFERMI, 1.000]
            header = lines[5].split()
            nedos = int(header[2])
            self.efermi = float(header[3])
            
            print(f"[INFO] DOSCAR Header: NEDOS={nedos}, E-Fermi={self.efermi:.4f}")
            
            # Total DOS 부분 건너뛰기 (header 6줄 + NEDOS 줄)
            current_line = 6 + nedos
            
            # 각 원자별 PDOS 읽기
            # 원자별 섹션: 헤더 1줄 + NEDOS 줄
            self.energies = []
            site_dos_list = []
            
            # 데이터 파싱 시작
            # 첫 번째 원자부터 끝까지
            atom_idx = 0
            while current_line < len(lines):
                # 원자 섹션 헤더 스킵
                current_line += 1 
                if current_line >= len(lines): break
                
                atom_dos = []
                temp_energies = []
                
                for _ in range(nedos):
                    vals = [float(x) for x in lines[current_line].split()]
                    # vals[0]: Energy
                    # vals[1:]: s_up, s_down, p_up, p_down, ... (or non-spin: s, p, d)
                    
                    if atom_idx == 0:
                        temp_energies.append(vals[0])
                        
                    # 모든 오비탈(s, p, d...) 합산
                    # Spin 유무에 따라 컬럼 수가 다르지만, 에너지(0번) 제외하고 다 더하면 Total DOS가 됨
                    total_at_energy = sum(vals[1:])
                    atom_dos.append(total_at_energy)
                    
                    current_line += 1
                
                if atom_idx == 0:
                    self.energies = np.array(temp_energies)
                
                site_dos_list.append(np.array(atom_dos))
                atom_idx += 1
                
            self.site_dos = site_dos_list
            print(f"[INFO] DOSCAR Loaded: Parsed {len(self.site_dos)} atoms.")
            
        except Exception as e:
            print(f"[ERROR] Failed to read DOSCAR: {e}")


def plot_ldos_from_files(poscar='POSCAR', doscar='DOSCAR', 
                         sigma=1.0, resolution=200, elim=(-2, 2)):
    """
    POSCAR와 DOSCAR를 사용하여 Z축 방향의 LDOS 히트맵을 그립니다.
    """
    # 1. 데이터 파싱
    parser = VaspParser(poscar_path=poscar, doscar_path=doscar)
    parser.read_poscar()
    parser.read_doscar()
    
    if not parser.z_coords.any() or not parser.site_dos:
        print("[ERROR] Data not loaded properly.")
        return

    # 2. 데이터 전처리
    energies = parser.energies - parser.efermi # Fermi Level 보정
    z_coords = parser.z_coords
    c_lattice = parser.c_lattice
    
    # Z축 그리드 생성
    z_grid = np.linspace(0, c_lattice, resolution)
    
    # LDOS 매트릭스 초기화 (Energy x Position)
    ldos_map = np.zeros((len(energies), len(z_grid)))
    
    # 3. Gaussian Smearing을 통한 LDOS 매핑
    # 기존 코드의 로직 유지: 각 원자의 DOS를 공간상에 가우시안 분포로 뿌림
    print("[PROCESSING] Calculating Spatial LDOS Map...")
    
    for i, z_pos in enumerate(z_coords):
        if i >= len(parser.site_dos): break # DOSCAR 원자 수가 POSCAR보다 적을 경우 방지
        
        dos_values = parser.site_dos[i]
        
        # 가우시안 가중치 계산 (Z축 방향)
        # exp( -0.5 * ((z - z_atom) / sigma)^2 )
        spatial_weight = np.exp(-0.5 * ((z_grid - z_pos) / sigma)**2)
        
        # Outer product로 효율적인 매트릭스 연산
        # (Energies, 1) x (1, Z_grid) -> (Energies, Z_grid)
        ldos_map += np.outer(dos_values, spatial_weight)

    # 4. 시각화 (이미지 스타일 모사)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    X, Y = np.meshgrid(z_grid, energies)
    
    c = ax.pcolormesh(X, Y, ldos_map, cmap='plasma', shading='gouraud', vmin=0, vmax=50)
    ax.axhline(0, 0, 1, c='gray', ls='--')
    
    ax.set_xlabel(r'Direction ($\mathrm{\AA}$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_ylim(elim)
    ax.set_xlim(0, c_lattice)
    
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('LDOS (arb. units)', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('LDOS.png', dpi=100)
    #plt.show()

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # sigma: Z축 스미어링 너비 (값이 클수록 부드럽게 퍼짐, 보통 1.0~2.0 추천)
    plot_ldos_from_files(poscar='POSCAR', doscar='DOSCAR', sigma=1.0, elim=(-5, 3))
