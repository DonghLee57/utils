import numpy as np
import matplotlib.pyplot as plt
import sys
fs = 12

filename = sys.argv[1]

fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)

data = np.loadtxt(filename).T

ax.plot(data[0], data[1])

ax.set_xlim(left=1.0, right=6.0)
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r'Distance ($\mathrm{\AA}$)',fontsize=fs)
ax.set_ylabel('RDF',fontsize=fs)
plt.show()
