import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plot = True

def implicit_equation(x, y):
    return x**2 + y**2 - 1

CURRENT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

data = np.load(os.path.join(DATA_DIR, "DataN200_theta1_+-180_theta2_0-180.npy"))

X = data[:, 2:4]  # Position
y = data[:, 0:2]  # Thetas

if plot:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot workspace
    axs[0].scatter(X[:, 0], X[:, 1], marker='.')
    axs[0].set_xlabel(r'$X$')
    axs[0].set_ylabel(r'$Y$')
    axs[0].grid(True)
    axs[0].axis('equal')

    # Plot workspace
    axs[1].scatter(y[:, 0], y[:, 1], marker='.')
    axs[1].axhline(y = np.pi, color='r')  # Add the horizontal line
    axs[1].axhline(y = -np.pi, color='r')  # Add the horizontal line
    axs[1].axhline(y = 2*np.pi, color='r')  # Add the horizontal line
    axs[1].axhline(y = -2*np.pi, color='r')  # Add the horizontal line
    axs[1].axvline(x = np.pi, color='r')  # Add the horizontal line
    axs[1].axvline(x = -np.pi, color='r')  # Add the horizontal line
    axs[1].axvline(x = 2*np.pi, color='r')  # Add the horizontal line
    axs[1].axvline(x = -2*np.pi, color='r')  # Add the horizontal line
    axs[1].set_xlabel(r'$\theta_1$')
    axs[1].set_ylabel(r'$\theta_2$')
    axs[1].grid(True)
    axs[1].axis('equal')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plots
    plt.show()