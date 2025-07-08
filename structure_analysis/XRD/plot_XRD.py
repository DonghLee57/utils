import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

def _calculate_ticks(min_val, max_val, num_ticks=5):
    """
    Calculate approximately num_ticks 'nice' tick positions for the interval
    (min_val, max_val), always ensuring that the endpoint (max_val) is included.

    Args:
        min_val (float): The minimum value of the axis range.
        max_val (float): The maximum value of the axis range.
        num_ticks (int): Desired number of ticks (default is 5).

    Returns:
        np.ndarray: Array of tick positions, including max_val as the rightmost tick.
    """
    rng = max_val - min_val
    if rng == 0:
        return [min_val]
    
    # Estimate a step size for the ticks based on the target number of ticks
    rough_step = rng / (num_ticks - 1)

    # Choose a 'nice' step size from a predefined set (e.g., 1, 2, 5, 10, 20, etc.)
    nice_steps = np.array([1, 2, 5, 10, 20, 25, 50])

    # Select the step size that is closest to the estimated step
    best_step = nice_steps[np.argmin(np.abs(nice_steps - rough_step))]

    # Determine the starting tick, rounded up to the nearest multiple of best_step
    start_tick = np.ceil(min_val / best_step) * best_step

    # Generate the tick positions up to, but not including, max_val
    ticks = np.arange(start_tick, max_val, best_step)

    # Ensure that max_val is included as the rightmost tick
    if max_val not in ticks:
        ticks = np.append(ticks, max_val)
    return ticks

def plot_multi_xrd_with_hkl(
    poscar_files,
    wavelength='CuKa',
    two_theta_range=(10, 90),
    fontsize_xticks=12,
    fontsize_yticks=12,
    fontsize_labels=14,
    fontsize_hkl=14,
    pad_xticks=10,
    pad_yticks=10
):
    """
    Calculate and plot XRD patterns from multiple VASP structure files (POSCAR format),
    displaying hkl indices in a top panel. Font sizes for x/y ticks, axis labels,
    and hkl indices, as well as tick label padding and fixed y-ticks, are all customizable.

    Args:
        poscar_files (list): List of file paths to POSCAR files to be analyzed.
        wavelength (str): X-ray wavelength source (e.g., 'CuKa', 'MoKa').
        two_theta_range (tuple): Tuple specifying the 2-theta range to plot (min, max).
        fontsize_xticks (int): Font size for x-axis tick labels.
        fontsize_yticks (int): Font size for y-axis tick labels.
        fontsize_labels (int): Font size for axis labels and plot title.
        fontsize_hkl (int): Font size for hkl index labels in the top panel.
        pad_xticks (int): Padding (in points) between x-axis ticks and the axis.
        pad_yticks (int): Padding (in points) between y-axis ticks and the axis.
    """
    # Create a figure with two vertically stacked subplots:
    # The top panel is for hkl indices, the main panel is for the XRD pattern.
    fig, (ax_top, ax_main) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        gridspec_kw={'height_ratios': [1, 4]}, # Set the height ratio for the two panels
        sharex=True # Share the x-axis between the panels
    )

    # Use a categorical color palette for plotting multiple patterns
    colors = plt.get_cmap('tab10').colors
    # Initialize the XRDCalculator with the specified wavelength and symmetry precision
    xrd_calculator = XRDCalculator(wavelength=wavelength, symprec=0.1)

    # Loop through each POSCAR file and plot its XRD pattern and hkl indices
    for i, poscar_file in enumerate(poscar_files):
        try:
            # Load the crystal structure from the POSCAR file
            structure = Structure.from_file(poscar_file)
            # Compute the XRD pattern for the given structure and 2-theta range
            pattern = xrd_calculator.get_pattern(structure, two_theta_range=two_theta_range)
            # Select a color for this pattern based on its index
            color = colors[i % len(colors)]
            # Use the file name as the legend label
            label = os.path.basename(poscar_file)
            # Normalize the intensity values so the maximum is 100
            intensities = pattern.y / max(pattern.y) * 100 if max(pattern.y) > 0 else pattern.y

            # Plot the XRD pattern as a stem plot on the main panel
            markerline, stemlines, baseline = ax_main.stem(
                pattern.x,
                intensities,
                linefmt='-',
                markerfmt='o',
                basefmt=' ',
                label=label
            )
            
            # Set the color and transparency for the stem lines and markers
            plt.setp(stemlines, 'color', color, alpha=0.7)
            plt.setp(markerline, 'color', color, 'markersize', 4)

            # For each peak, add the corresponding hkl indices as text in the top panel
            for two_theta, hkl_group in zip(pattern.x, pattern.hkls):
                # Concatenate multiple hkl indices for the same peak, separated by commas
                hkl_str = ', '.join([
                    f"({hkl_dict['hkl'][0]}{hkl_dict['hkl'][1]}{hkl_dict['hkl'][2]})" for hkl_dict in hkl_group
                ])

                # Place the hkl text at the peak position, rotated 90 degrees for readability
                ax_top.text(
                    two_theta, 0.1, hkl_str, color=color,
                    rotation=90, verticalalignment='bottom',
                    horizontalalignment='center', fontsize=fontsize_hkl
                )
        except Exception as e:
            # Print an error message if the file could not be processed
            print(f"Error processing file '{poscar_file}': {e}")
            continue

    # --- Final plot styling ---
    # Configure the top panel (hkl indices)
    ax_top.set_ylim(0, 1)
    ax_top.axis('off') # Hide axes and borders for a clean look

    # Configure the main panel (XRD patterns)
    ax_main.set_xlim(two_theta_range)
    ax_main.set_ylim(0, 100)

    # Calculate and set x-ticks, ensuring the right endpoint is included
    xticks = _calculate_ticks(two_theta_range[0], two_theta_range[1])
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels([str(tick) for tick in xticks], fontsize=fontsize_xticks)

    # Set fixed y-ticks at [0, 25, 50, 75, 100] and apply font size
    yticks = [0, 25, 50, 75, 100]
    ax_main.set_yticks(yticks)
    ax_main.set_yticklabels([str(y) for y in yticks], fontsize=fontsize_yticks)

    # Add padding between tick labels and axes for better readability
    ax_main.tick_params(axis='x', pad=pad_xticks)
    ax_main.tick_params(axis='y', pad=pad_yticks)

    # Set axis labels, grid, and legend
    ax_main.set_xlabel(r"2$\theta$ (degrees)", fontsize=fontsize_labels)
    ax_main.set_ylabel("Normalized Intensity (a.u.)", fontsize=fontsize_labels)
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.legend(fontsize=fontsize_labels)

    # Set the main title for the entire figure
    fig.suptitle(f"XRD Patterns ({wavelength})", fontsize=fontsize_labels + 4, y=0.95)

    # Adjust layout to prevent title overlap and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # List of POSCAR files to plot
    poscar_list = ["Si.poscar", "TiN.poscar"]

    # Call the function with custom font sizes
    plot_multi_xrd_with_hkl(
        poscar_list, 
        two_theta_range=(20, 60),
        fontsize_xticks=20,      # Font size for x-axis ticks
        fontsize_yticks=20,      # Font size for y-axis ticks
        fontsize_labels=20,      # Font size for axis labels and legend
        fontsize_hkl=20,         # Font size for hkl indices
        pad_xticks=10,           # Increase horizontal tick label spacing
        pad_yticks=10            # Increase vertical tick label spacing
)
