import numpy as np
import os
import sys
import matplotlib.pyplot as plt

CSD = os.path.dirname(__file__)
BSD = os.path.dirname(CSD)
DATA_FOLDER = os.path.join(BSD, "data")

# Theta v Phi
Data_phi = np.load(os.path.join(DATA_FOLDER, "DataN1000_phi.npy"))
theta_p = Data_phi[:,0]
phi_p = Data_phi[:,1]

print(f"Size of theta_p: {theta_p.shape}")
print(f"Size of phi_p: {phi_p.shape}")

# Theta vs (X,Y) -- coupler curve
Data_XY = np.load(os.path.join(DATA_FOLDER, "DataN1000_XY.npy"))
theta_xy = Data_XY[:,0]
XY_xy = Data_XY[:,1:3]

print(f"Size of theta_xy: {theta_xy.shape}")
print(f"Size of XY_xy: {XY_xy.shape}")

# Plotting
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.scatter(theta_p, phi_p, marker='.')
plt.xlabel(r'$\theta_1$ (rads)')
plt.ylabel(r'$\phi_1$ (rads)')
plt.title(r'$\phi_1$ vs $\theta_1$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(XY_xy[:,0], XY_xy[:,1], marker='.')
plt.xlabel(r'X')
plt.ylabel(r'Y')
plt.title(r'Coupler curve')
plt.axis('equal')
plt.grid(True)

# Adjust layout
plt.tight_layout()

plt.show()