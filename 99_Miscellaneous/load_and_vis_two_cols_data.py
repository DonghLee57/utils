import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = sys.argv[1]
    data = read_data(filename, min_col=0.0, max_col=100.0)

    col_index = 0  # col{col_index}
    col1, col2 = get_profile(data, col_index)
    fn = create_interpolator(col1, col2)

    ss = [0.0, 0.5, 1.0]
    fig, ax = plt.subplots(1,1, figsize=(8,5), constrained_layout=True)
    ax.plot(col1, col2, label=f'Raw')
    for i, s in enumerate(ss):
        profiles, points = create_points(col1, col2, num_regions=15, sensitivity=s)
        for j, pt in enumerate(points): 
            x = points[j:j+2]
            if j == 0: 
                ax.plot(x, fn(x) ,c=f'C{i+1}', label=f'sensitivity={s:.1f}')
            else:
                ax.plot(x, fn(x) ,c=f'C{i+1}')
    ax.set_xlabel(r'Col 1')
    ax.set_ylabel(r'Col 2')
    ax.legend()
    plt.show()

# Functions
def read_data(filename, max_col=None, min_col=None, skip_header=1):
    data = np.genfromtxt(filename, skip_header=skip_header)
    col = data[:, 0]
    mask = np.ones(len(data), dtype=bool)
    if max_time is not None:
        mask = mask & (col <= max_col)
    if min_time is not None:
        mask = mask & (col >= min_col)
    filtered_data = data[mask]
    return filtered_data

def get_profile(data, col_index):
    col1 = data[:, 0]
    col2 = data[:, col_index + 1]
    return col1, col2

def create_points(col1, col2, num_regions=10, sensitivity=None):
    if sensitivity == None:
        points = np.linspace(col1.min(), col1.max(), num_regions + 1)
        profiles = []
        return profiles, points
    else:
        col2_diff = np.abs(np.diff(col2))
        col1_diff = np.diff(col1)
        col1_diff[col1_diff == 0] = np.min(col1_diff[col1_diff > 0]) * 0.01
        rate = col2_diff / col1_diff
        weights = np.log1p(rate)
        weights = weights ** sensitivity
        
        cum_weights = np.zeros(len(col1))
        cum_weights[1:] = np.cumsum(weights)
        total_weight = cum_weights[-1]
        
        target_weights = np.linspace(0, total_weight, num_regions + 1)

        points = []
        for target in target_weights:
            idx = np.argmin(np.abs(cum_weights - target))
            points.append(col1[idx])
        points = np.array(points)
        points = np.unique(points)

        while len(points) < num_regions + 1:
            intervals = np.diff(points)
            max_interval_idx = np.argmax(intervals)
            new_point = (points[max_interval_idx] + points[max_interval_idx + 1]) / 2
            points = np.sort(np.append(points, new_point)
                                  
        return profiles, points

def create_interpolator(col1, col2):
    sort_idx = np.argsort(col1)
    col1_sorted = col1[sort_idx]
    col2_sorted = col2[sort_idx]
    def interpolator(x):
        return np.interp(x, col1_sorted, col2_sorted)
    return interpolator

# End
if __name__ == "__main__":
    main()
