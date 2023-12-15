import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plot = False

CSD = os.path.dirname(__file__)
DATAGEN_DIR = os.path.join(CSD, "data")

L1 = 5
L2 = 3

N = 200

ang = np.linspace(0, 2 * np.pi, N)
a = ang.reshape(N, 1)

b = a

A, B = np.meshgrid(a, b)

A_flat = A.flatten()
B_flat = B.flatten()

theta_1 = A_flat.reshape(N * N, 1)
theta_2 = B_flat.reshape(N * N, 1)

X = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
Y = L1 * np.sin(theta_1) + L2 * np.sin(theta_1 + theta_2)

np.save(os.path.join(DATAGEN_DIR, 'DataN200.npy'), [theta_1, theta_2, X, Y])

if plot:
    plt.scatter(X, Y, marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
