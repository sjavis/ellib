#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

init = np.loadtxt("coords.txt")

# View minima
up = np.loadtxt("buckled_up.txt")
down = np.loadtxt("buckled_down.txt")
ts = np.loadtxt("transition_state.txt")

plt.plot(init[:,0], init[:,2], 'k.')
plt.plot(up[:,0], up[:,2], 'r.')
plt.plot(down[:,0], down[:,2], 'b.')
plt.plot(ts[:,0], ts[:,2], 'g.')

plt.gca().set_aspect('equal')
plt.show()
