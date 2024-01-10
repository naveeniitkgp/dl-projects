import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plot = False

CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
DATAGEN_DIR = os.path.join(BDR, "data")

parser = argparse.ArgumentParser(description='Data_gen Parameters')
parser.add_argument('--n', type=int, default=200, help='Number of data points')
parser.add_argument('--t1_min', type=float, default=0, help='Min value of theta_1 (degress)')
parser.add_argument('--t2_min', type=float, default=0, help='Min value of theta_2 (degress)')
parser.add_argument('--t1_max', type=float, default=360, help='Max value of theta_1 (degress)')
parser.add_argument('--t2_max', type=float, default=360, help='Max value of theta_2 (degress)')
parser.add_argument('--data_name', type=str, default='#######', help='Data Folder Name')
args = parser.parse_args()

DATA_FOLDER_NAME = os.path.join(DATAGEN_DIR, f"{args.data_name}")
os.makedirs(DATA_FOLDER_NAME, exist_ok=True)

# python src\data_gen.py --n 100 --t1_min -180 --t1_max 180 --t2_min -180 --t2_max 180 --data_name "N100_theta12_+-180"

L1 = 5
L2 = 3

N = args.n

a = np.deg2rad(np.linspace(args.t1_min, args.t1_max, N))
b = np.deg2rad(np.linspace(args.t2_min, args.t2_max, N))

A, B = np.meshgrid(a, b)

theta_1 = A.flatten()
theta_2 = B.flatten()

X = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
Y = L1 * np.sin(theta_1) + L2 * np.sin(theta_1 + theta_2)

stacked_data = np.stack((theta_1, theta_2, X, Y), axis=1)

np.save(os.path.join(DATA_FOLDER_NAME, 'main_data.npy'), stacked_data)

if plot:
    plt.scatter(X, Y, marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Creating a fixed Training and Test dataset
# Input and output for inverse problem
X = stacked_data[:, 2:4]  # Position
y = stacked_data[:, 0:2]  # Thetas

tr_ratio = 0.85
ts_ratio = 1 - tr_ratio

data_train, data_test = train_test_split(stacked_data, test_size=ts_ratio, shuffle=True, random_state=143)

print(data_train[0])
print(data_train.shape)

np.save(os.path.join(DATA_FOLDER_NAME, 'train_data.npy'), data_train)
np.save(os.path.join(DATA_FOLDER_NAME, 'test_data.npy'), data_test)

