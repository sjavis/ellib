#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def energy(x, y):
    peaks = np.array([
      [-3, -1.4, 0, 1, 1],
      [-2,  1.4, 0, 1, 1],
      [-1, 0.07, 1, 1, 1]])
    e = 0
    for peak in peaks:
        dx = (x - peak[1]) / peak[3]
        dy = (y - peak[2]) / peak[4]
        e += peak[0] * np.exp(-dx**2 - dy**2);
    return e


def plot_energy(ax=plt.gca()):
    x, y = np.mgrid[-2:2:100j, -0.8:1.3:100j]
    e = energy(x, y)
    ax.contourf(x, y, e, cmap='gray')
    ax.set_aspect('equal')

def plot_path():
    path = np.loadtxt('path.txt')
    plt.plot(path[:,0], path[:,1], c='r', marker='s', mec='k')

plot_energy()
plot_path()
plt.show()
