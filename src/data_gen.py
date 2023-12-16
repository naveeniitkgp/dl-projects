import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plot = False

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
DATAGEN_DIR = os.path.join(BDR, "data")

L1 = 5
L2 = 3

N = 200

ang = np.linspace(0, 2 * np.pi, N)
a = ang.reshape(N, 1)

b = a

A, B = np.meshgrid(a, b)

theta_1 = A.flatten()
theta_2 = B.flatten()

X = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
Y = L1 * np.sin(theta_1) + L2 * np.sin(theta_1 + theta_2)

stacked_data = np.stack((theta_1, theta_2, X, Y), axis=1)

np.save(os.path.join(DATAGEN_DIR, 'DataN200.npy'), stacked_data)

if plot:
    plt.scatter(X, Y, marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
