import os
import sys
import argparse
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

CSD = os.path.dirname(__file__)
BSD = os.path.dirname(CSD)
DATA_FOLDER = os.path.join(BSD, "data")

parser = argparse.ArgumentParser(description='Data_gen Parameters')
parser.add_argument('--n', type=int, default=1000, help='Number of data points')
parser.add_argument('--tr_ratio', type=float, default=0.85, help='Training data split ratio')
args = parser.parse_args()

N = args.n
# Define the lengths
L1 = 1
L2 = 3
L3 = 4
L0 = 5

animate_4bar = False  # To animate the motion

# Enable LaTeX fonts
rc('text', usetex=True)
rc('font', family='serif')


DATA_FOLDER_NAME = os.path.join(DATA_FOLDER, f"data_{N}")
os.makedirs(DATA_FOLDER_NAME, exist_ok=True)

def equation(phi_1, theta_1):
    return L0**2 + L1**2 + L3**2 - L2**2 - 2 * L0 * L1 * np.cos(theta_1) - np.cos(phi_1) * (2 * L1 * L3 * np.cos(theta_1) - 2 * L0 * L3) - np.sin(phi_1) * 2 * L1 * L3 * np.sin(theta_1)


initial_guess_value = 1

# Example run for one value
# theta_1_input = np.pi/6  # Replace with the actual value

# phi_1_output = fsolve(equation, initial_guess_value, args=(theta_1_input,))
# print(f"For theta_1 = {theta_1_input}, phi_1 is approximately {phi_1_output[0]}")

# Example to plot the result
phi_1_values = np.linspace(0, 2 * np.pi, N)
theta_1_values_1 = [fsolve(equation, initial_guess_value, args=(theta_1,))[0] for theta_1 in phi_1_values]
theta_1_values_2 = [fsolve(equation, -initial_guess_value, args=(theta_1,))[0] for theta_1 in phi_1_values]

theta1 = np.concatenate([phi_1_values, phi_1_values])  # Input - Making two copies of theta_1 one below another
phi = np.concatenate([theta_1_values_1, theta_1_values_2])  # Output
# Data_phi = np.column_stack((theta1, phi))

# np.save(os.path.join(DATA_FOLDER_NAME, 'main.npy'), Data_phi)

tr_ratio = args.tr_ratio
ts_ratio = 1 - tr_ratio

# data_train_p, data_test_p = train_test_split(Data_phi, test_size=ts_ratio, shuffle=True, random_state=143)

# np.save(os.path.join(DATA_FOLDER_NAME, 'train_data.npy'), data_train_p)
# np.save(os.path.join(DATA_FOLDER_NAME, 'test_data.npy'), data_test_p)

# print(f"FOR THETA vs PHI")
# print(f"Size of theta1: {theta1.shape}")
# print(f"Size of phi: {phi.shape}")
# print(f"Size of Data_phi: {Data_phi.shape}")
# print(f"Size of Data_phi Train: {data_train_p.shape}")
# print(f"Size of Data_phi Test: {data_test_p.shape}")

# Getting each point of the four bar
C0 = np.zeros((N, 2))
C1 = np.column_stack((L1 * np.cos(phi_1_values), L1 * np.sin(phi_1_values)))
C3 = np.zeros((N, 2)) + [L0, 0]
C2_sol1 = C3 + np.column_stack((L3 * np.cos(theta_1_values_1), L3 * np.sin(theta_1_values_1)))
C2_sol2 = C3 + np.column_stack((L3 * np.cos(theta_1_values_2), L3 * np.sin(theta_1_values_2)))

# Getting a point on the coupler
xy_ratio = 0.5  # Ratio that determines the position of the point on the coupler
XY_sol1 = C1 + xy_ratio * (C2_sol1 - C1)
XY_sol2 = C1 + xy_ratio * (C2_sol2 - C1)

# IP_XY = theta1  # Making two copies -- same as theta1
XY = np.concatenate([XY_sol1, XY_sol2])
Data_XY = np.column_stack((theta1, phi, XY))
np.save(os.path.join(DATA_FOLDER_NAME, 'main.npy'), Data_XY)

# tr_ratio = args.tr_ratio
# ts_ratio = 1 - tr_ratio

data_train_xy, data_test_xy = train_test_split(Data_XY, test_size=ts_ratio, shuffle=True, random_state=143)

np.save(os.path.join(DATA_FOLDER_NAME, 'train_data.npy'), data_train_xy)
np.save(os.path.join(DATA_FOLDER_NAME, 'test_data.npy'), data_test_xy)

print(f"FOR THETA vs (X,Y)")
print(f"Size of theta1: {theta1.shape}")
print(f"Size of XY: {XY.shape}")
print(f"Size of Data_XY: {Data_XY.shape}")
print(f"Size of Data_XY Train: {data_train_xy.shape}")
print(f"Size of Data_XY Test: {data_test_xy.shape}")

if animate_4bar:

    def update(frame):
        plt.clf()  # Clear the current plot
        colors = ['red', 'green', 'blue', 'purple']
        for i in range(4):  # Loop through each line
            color = colors[i]
            plt.plot([C0[frame, 0], C1[frame, 0], C2_sol1[frame, 0], C3[frame, 0], C0[frame, 0]][i:i + 2],
                     [C0[frame, 1], C1[frame, 1], C2_sol1[frame, 1], C3[frame, 1], C0[frame, 1]][i:i + 2],
                     'o-', color=color)
        plt.grid(True)
        plt.title(f'Frame {frame}/{N-1}')
        plt.xlim(-2, 6)
        plt.ylim(-4, 4)

    # Create the initial plot
    fig, ax = plt.subplots()
    ax.plot([], [], 'o-')  # Empty plot to be updated in each frame

    # Animation settings
    animation = FuncAnimation(fig, update, frames=N, interval=10, repeat=False)

    # Show the animation
    plt.show()
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(phi_1_values, theta_1_values_1, label=f'Solution 1')
plt.plot(phi_1_values, theta_1_values_2, label=f'Solution 2')
plt.xlabel(r'$\theta_1$ (rads)')
plt.ylabel(r'$\phi_1$ (rads)')
plt.title(r'$\phi_1$ vs $\theta_1$')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(phi_1_values, np.linalg.norm(C1 - C0, axis=1), label=f'Link 1')
plt.plot(phi_1_values, np.linalg.norm(C1 - C2_sol1, axis=1), label=f'Link 2')
plt.plot(phi_1_values, np.linalg.norm(C2_sol1 - C3, axis=1), label=f'Link 3')
plt.plot(phi_1_values, np.linalg.norm(C3 - C0, axis=1), label=f'Link 0')
plt.xlabel(r'$\theta_1$ (rads)')
plt.ylabel(r'Link lengths')
plt.title(r'Link lengths vs $\theta_1$')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(XY_sol1[:, 0], XY_sol1[:, 1], label=f'Solution 1')
plt.plot(XY_sol2[:, 0], XY_sol2[:, 1], label=f'Solution 2')
# plt.scatter(Data_XY[:,1], Data_XY[:,2], label=f'Solution 2')
plt.xlabel(r'X')
plt.ylabel(r'Y')
plt.title(r'Coupler curve')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Adjust layout
plt.tight_layout()

plt.show()
