import numpy as np
import matplotlib.pyplot as plt
import sys
fs = 12

filename = sys.argv[1]

fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)

data = np.loadtxt(filename).T

ax.plot(data[0], data[1], label='legend')

ax.set_title('Title', fontsize=fs*1.2)

ticks = np.arange(1,7,1)
ax.set_xticks(ticks)
ax.set_xticklabels([f'{x:.0f}' for x in ticks], fontsize=fs)
ax.set_xlim([ticks[0],ticks[-1]])
ax.set_xlabel(r'Distance ($\mathrm{\AA}$)',fontsize=fs)

ticks = np.arange(0,13,2.5)
ax.set_yticks(ticks)
ax.set_yticklabels([f'{x:.1f}' for x in ticks], fontsize=fs)
ax.set_ylim([ticks[0],ticks[-1]])
ax.set_ylabel('RDF',fontsize=fs)

ax.legend(fontsize=fs)
plt.show()
