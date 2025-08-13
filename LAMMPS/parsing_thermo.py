import glob
import re 
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def main():
    log_file = 'lammps.log' 
    thermo_data = parse_lammps_log(f'{log_file}')

    # Examples
    base_path = './'
    fig, ax = plot_thermo_step(thermo_data, 
                               'Temp', 
                               save_path=f"{base_path}/temp_plot.png")
    _, ax2 = plot_thermo_step(thermo_data, 
                              'TotEng', 
                              fig=fig, 
                              ax=ax,
                              right_ax=True,
                              line_args={'color': 'red', 'ls': '--'},
                              save_path=f"{base_path}/temp_toteng_plot.png")
    return 0

def parse_lammps_log(log_path: str) -> dict:
    """
    Parses a LAMMPS log file to extract thermodynamic data and the number of atoms.

    This function reads a LAMMPS log file, finds thermodynamic output
    sections (indicated by a 'Step' header), and extracts the data.
    It correctly handles duplicate steps that occur across different
    simulation stages by overwriting with the last seen value for that step.
    The number of atoms is extracted from the 'Loop time' line.

    Args:
        log_path: The path to the LAMMPS log file.

    Returns:
        A dictionary where keys are thermodynamic property names (str) and
        values are 2D NumPy arrays of shape (N, 2), with columns for step
        and value. It also includes a 'Natoms' key with the number of atoms (int).
    """
    intermediate_data = defaultdict(dict)
    in_thermo_section = False
    headers = []
    n_atoms = None  # To store the number of atoms if found

    # Regex patterns for parsing
    header_pattern = re.compile(r"^\s*Step\s+.*")
    end_pattern = re.compile(r"^\s*Loop time of.*")

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for the start of a new thermo block
            if header_pattern.match(line):
                in_thermo_section = True
                headers = line.strip().split()
                continue
            
            # Check for the end of the current thermo block
            if end_pattern.match(line):
                in_thermo_section = False
                headers = []
                try:
                    words = line.split()
                    if 'atoms' in words:
                        atoms_index = words.index('atoms')
                        n_atoms = int(words[atoms_index - 1])
                except (ValueError, IndexError):
                    # Silently ignore if the line format is unexpected
                    pass
                continue

            # Process data lines if inside a thermo section
            if in_thermo_section:
                try:
                    values = [float(v) for v in line.strip().split()]
                    if len(values) != len(headers):
                        continue
                    
                    current_step = int(values[0])
                    
                    # Associate values with headers, skipping the 'Step' column itself
                    for i, header in enumerate(headers[1:], 1):
                        intermediate_data[header][current_step] = values[i]

                except (ValueError, IndexError):
                    in_thermo_section = False
                    headers = []

    # Convert the intermediate dictionary to the final sorted list format
    final_data = {}
    if n_atoms is not None:
        final_data['NATOMS'] = n_atoms
    for keyword, step_value_map in intermediate_data.items():
        sorted_items = sorted(step_value_map.items())
        final_data[keyword] = np.array(sorted_items, dtype=np.float64)

    return final_data

def plot_thermo_step(
    thermo_dict,
    keyword,
    fig=None,
    ax=None,
    right_ax=False,
    save_path=None,
    line_args=None
):
    """
    Plot Step vs. thermo keyword data from LAMMPS parsing result.
    Optionally add extra curves to the secondary y-axis (right) of an existing figure.

    Args:
        thermo_dict (dict): {keyword: np.ndarray ((N,2)), ...}
        keyword (str): thermo property to plot on y-axis, x-axis is always Step
        fig (matplotlib.figure.Figure, optional): Figure to plot on (if None, new).
        ax (matplotlib.axes.Axes, optional): Axis to plot on (if None, new).
        right_ax (bool): If True, plot on right y-axis (secondary axis).
        save_path (str): If given, save the figure to this path.
        line_args (dict): Optional line formatting arguments (color, marker, etc.)

    Returns:
        tuple: (fig, ax) or (fig, ax2)
    """
    if keyword not in thermo_dict:
        print(f"'{keyword}' not found. Available keys: {list(thermo_dict.keys())}")
        return None, None

    arr = thermo_dict[keyword]
    steps, values = arr[:, 0], arr[:, 1]
    line_args = line_args or {}

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if right_ax:
        # add to the right y-axis as a new curve
        ax2 = ax.twinx()
        ax2.plot(steps, values, label=keyword, **line_args)
        ax2.set_ylabel(keyword)
        ax2.legend(loc='best')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig, ax2
    else:
        ax.plot(steps, values, label=keyword, **line_args)
        ax.set_xlabel('Step')
        ax.set_ylabel(keyword)
        ax.legend(loc='best')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig, ax

if __name__ == '__main__':
    main()
