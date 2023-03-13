#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, axes = plt.subplots(2, 1)
plt.sca(axes[0])

# Plot potential
x, y = np.mgrid[-1:2:20j, -1:2:20j]
efn = lambda x, y: x**2 + y**2
e = efn(x, y)
plt.contourf(x, y, e, cmap='Greys')

# Plot genetic algorithm
pop_size = 100
data = np.loadtxt("output.txt").reshape((-1, pop_size, 2))
cmap = plt.get_cmap("viridis", len(data))
sm = mpl.cm.ScalarMappable(mpl.colors.Normalize(0,len(data)), cmap)

for it, pop in enumerate(data):
    t = it / len(data)
    plt.plot(pop[:,0], pop[:,1], ls='none', marker='.', c=sm.to_rgba(it))
cbar = plt.colorbar(sm)
cbar.set_label('Generation No.')

# Plot best energy over time
plt.sca(axes[1])
ebest = np.zeros(len(data))
for it, pop in enumerate(data):
    epop = efn(pop[:,0], pop[:,1])
    plt.scatter(np.full_like(epop,it), epop, c='lightgray', s=2)
    ebest[it] = np.min(epop)
plt.plot(ebest)
plt.xlabel('Generation')
plt.ylabel('Energy')
plt.semilogy()

plt.tight_layout()
plt.show()
