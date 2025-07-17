import glob
import re 
from collections import defaultdict
import numpy as np

def main():
    log_file = 'lammps.log' 
    thermo_data = parse_lammps_log(f'{log_file')
    return 0

def parse_lammps_log(log_file_path: str) -> dict:
    """
    Parses a LAMMPS log file to extract thermodynamic data.

    This function reads a LAMMPS log file, finds thermodynamic output
    sections (indicated by a 'Step' header), and extracts the data.
    It correctly handles duplicate steps that occur across different
    simulation stages (e.g., after a 'minimize' followed by a 'run')
    by overwriting with the last seen value for that step.

    Args:
        log_file_path: The path to the LAMMPS log file.

    Returns:
        A dictionary where keys are the thermodynamic property names (str)
        and values are lists of [step, value] pairs, sorted by step.
        Example: 
        {
            'Temp': [[0, 0.0], [1000, 1661.95], ...],
            'TotEng': [[0, -960.65], [1000, -918.36], ...]
        }
    """
    # Use an intermediate dict to handle step overwrites easily.
    # The defaultdict simplifies initialization.
    # Format: {'Temp': {step1: val1, step2: val2}, ...}
    intermediate_data = defaultdict(dict)
    
    # State variables for parsing
    in_thermo_section = False
    headers = []

    # Regex to find the header line of a thermo output block
    header_pattern = re.compile(r"^\s*Step\s+.*")
    # Regex to find the end of a thermo block
    end_pattern = re.compile(r"^\s*Loop time of.*")

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # maybe inserted into the block (end_pattern=True)
            if 'atoms' in line.split() and len(line.split()) == 2: 
                n_atoms = int(line.strip().split()[0])

            # Check for the start of a new thermo block
            if header_pattern.match(line):
                in_thermo_section = True
                headers = line.strip().split()
                continue
            
            # Check for the end of the current thermo block
            if end_pattern.match(line):
                in_thermo_section = False
                headers = []
                continue

            # Process data lines if inside a thermo section
            if in_thermo_section:
                try:
                    # Split line and convert to numbers
                    values = [float(v) for v in line.strip().split()]
                    
                    # Ensure the number of values matches the number of headers
                    if len(values) != len(headers):
                        continue
                    
                    current_step = int(values[0])
                    
                    # Associate values with headers, skipping the 'Step' column itself
                    for i, header in enumerate(headers[1:], 1):
                        # This assignment overwrites any previous value for the same step,
                        # effectively handling duplicates.
                        intermediate_data[header][current_step] = values[i]

                except (ValueError, IndexError):
                    # If conversion fails or list is empty, it's not a data line.
                    # This can happen on blank lines or other info lines.
                    # We can optionally end the thermo section here as well.
                    in_thermo_section = False
                    headers = []

    # Convert the intermediate dictionary to the final sorted list format
    final_data = {}
    final_data['NATOMS'] = n_atoms
    for keyword, step_value_map in intermediate_data.items():
        # Sort items by step (the dictionary key) and convert to list of lists
        sorted_items = sorted(step_value_map.items())
        final_data[keyword] = [list(item) for item in sorted_items]
    return final_data

if __name__ == '__main__':
    main()
