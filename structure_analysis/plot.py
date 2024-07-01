import matplotlib.pyplot as plt
import sys
filename = sys.argv[1]

fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)

data = np.loadtxt(filename).T

ax.plot(data[0], data[1])

plt.show()
