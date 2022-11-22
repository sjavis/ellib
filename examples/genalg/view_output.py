#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Plot potential
x, y = np.mgrid[-1:2:20j, -1:2:20j]
e = x**2 + y**2
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

plt.tight_layout()
plt.show()
