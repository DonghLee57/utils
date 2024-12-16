import sys, glob
import numpy as np

# ex) ['./out_300', './out_400', './out_500']
outcars = glob.glob('./out*')
outcars.sort(key=lambda x: float(x.split('_')[-1]))
energy = np.zeros((len(outcars), 3))
pressure = np.zeros((len(outcars), 2))
for i in range(len(outcars)):
  tmp = open(outcars[i],'r')
  out = tmp.readlines()
  for idx, line in enumerate(out):
    items = line.split()
    if 'NIONS'   in items:
      natom = int(items[-1])
    elif 'TOTEN' in items:
      total_energy= float(items[-2])
    elif 'pressure' in items:
      press = float(items[-7])

  energy[i] = [total_energy, total_energy/natom, 0]
  pressure[i] = [press, 0]

energy[:,2] = energy[:,1] - energy[-1][1]
pressure[:,1] = pressure[:,0] - pressure[-1][0]
fe = open('energy.dat','w')
fp = open('press.dat','w')

for idx, out in enumerate(outcars):
  fe.write(f'{out:14s} {energy[idx][0]:18.4f} {energy[idx][1]:8.4f} {energy[idx][2]:8.4f}\n')
  fp.write(f'{out:14s} {pressure[idx][0]:10.4f} {pressure[idx][1]:10.4f}\n')
