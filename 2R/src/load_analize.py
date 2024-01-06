import torch
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score

# parserD = argparse.ArgumentParser(description='Enter model name')
# parserD.add_argument('--results_folder', type=str, default='ik_nn_03-01-24_mod_inv_v1', help='Name of the folder')
# argsD = parserD.parse_args()

L1 = 5
L2 = 3
N = 1000 # Number of sampling points
RUN_NAME = 'ik_nn_03-01-24_mod_inv_v11' # Name of the folder inside _result
colors = np.linspace(0, 1, N)  # Gradient of colors

# Add the RESULTS_FOLDER to the system path
CSD = os.path.dirname(__file__)
BDR = os.path.dirname(CSD)
RESULTS_FOLDER = os.path.join(BDR, "_results", RUN_NAME)
MODEL_FOLDER = os.path.join(RESULTS_FOLDER, "ckpt_in_train")

# Make sure RESULTS_FOLDER is added before importing any local modules
sys.path.insert(0, RESULTS_FOLDER)

# Import the inv2r_nn from RESULTS_FOLDER
from nn_inv_2r import inv2r_nn

# Get a list of all files in the MODEL_FOLDER
files = os.listdir(MODEL_FOLDER)

# Filter the list to include only .pth files
pth_files = [file for file in files if file.endswith(".pth")]

# Ensure there is at least one .pth file
if not pth_files:
    raise FileNotFoundError("No .pth files found in the specified folder.")

# Get the full path of the last modified .pth file
latest_pth_file = max(pth_files, key=lambda x: os.path.getmtime(os.path.join(MODEL_FOLDER, x)))

MODEL_NAME = latest_pth_file # Best model inside the folder

# Configure matplotlib to use LaTeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def linear_model(x, m, c):
    return m * x + c

def IK2R(X,Y):
    theta_2 = np.arccos((X**2 + Y**2 - L1**2 - L2**2)/(2*L1*L2))
    theta_1 = np.arctan2(Y,X) - np.arctan2(L2*np.sin(theta_2),(L1 + L2*np.cos(theta_2)))

    return theta_1, theta_2

def FK2R(theta_1, theta_2):
    X = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
    Y = L1 * np.sin(theta_1) + L2 * np.sin(theta_1 + theta_2)

    return X, Y

def load_model(model_path):
    model = inv2r_nn().to('cpu')  # Assuming the model is to be run on CPU
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['net_state_dict'])
    model.eval()
    return model

def predict(model, input_data):
    with torch.no_grad():
        prediction = model(input_data)
        return prediction.numpy()

def load_and_plot_models(model_folder, input_data):
    # Extract epoch numbers from filenames using regular expression
    MODEL_PATH = os.path.join(model_folder, MODEL_NAME)

    # Load the model
    model = load_model(MODEL_PATH)

    # Get the prediction for the input data
    output = predict(model, input_data)

    # Flatten and squeeze the output for compatibility with plt.plot()
    return np.squeeze(output)

# Plotting functions

def plot_results(X_data, Y_data, theta_data, r_data, X_pred_data, Y_pred_data, theta_1_data, theta_2_data, C2_data, x_ideal_data, y_ideal_data, C2_fit_data, X2Y2_fit_data, r2X_data, r2Y_data, spread_text):
    # PLOT
    plt.figure(figsize=(12, 8))

    # Plot the reachable workspace
    plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
    plot_generated_points()

    # Plot the distribution for r and theta
    plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    plot_2r_ik()

    # Plot X^2 + Y^2 = 2*L1*L2*cos(theta2) + L1^2 + L2^2
    plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), X_data, X_pred_data, r'Parity plot for X $\left( R^2 = {:.4f} \right)$'.format(r2X_data))

    # Plot parity plot for Y
    plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), Y_data, Y_pred_data, r'Parity plot for Y $\left( R^2 = {:.4f} \right)$'.format(r2Y_data))

    # Plot the spread of theta1 and theta2
    plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)
    plot_theta_spread()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
def plot_generated_points():
    plt.scatter(X, Y, marker='.', c=colors, cmap='viridis')
    plt.quiver(X,Y,(X_pred-X),(Y_pred-Y), angles='xy', scale_units='xy', scale=1, color='k', width=0.004, headlength=0, headaxislength=0)
    plt.plot((L1+L2)*np.cos(ang), (L1+L2)*np.sin(ang), color='r')
    plt.plot((L1-L2)*np.cos(ang), (L1-L2)*np.sin(ang), color='r')
    plt.xlabel(r'X')
    plt.ylabel(r'Y')
    plt.title(r'Generated points')
    plt.grid(True)
    plt.axis('equal')

def plot_distributions():
    plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
    sns.histplot(r, kde=True, color='c', bins=25)
    plt.title(r'Distribution of $r$')

    plt.subplot2grid((4, 6), (1, 2), colspan=2, rowspan=1)
    sns.histplot(theta, kde=True, color='m', bins=25)
    plt.title(r'Distribution of $\theta$')

def plot_2r_ik():
    plt.scatter(np.cos(theta_2), (X**2 + Y**2), marker='.', c=colors, cmap='viridis', alpha=0.5, label=f'Test Points')
    plt.plot(x_ideal, y_ideal, color='r', linestyle='-.', label=r'Ideal: $m = {:.2f}$, $c = {:.2f}$'.format(m_ideal, c_ideal))
    plt.plot(C2_fit, X2Y2_fit, color='b', linestyle='--', label=r'Best-fit: $m = {:.2f}$, $c = {:.2f}$'.format(m_fit, c_fit))
    plt.xlabel(r'$\cos \left( \theta_2 \right)$')
    plt.ylabel(r'$X^2 + Y^2$')
    plt.grid(True)
    plt.legend()
    plt.title(r'2R IK : $X^2 + Y^2 = 2 l_1 l_2 \cos \left( \theta_2 \right) + l_1^2 + l_2^2$')

def plot_parity_plot(ax, actual, predicted, label):
    ax.scatter(actual, predicted, alpha=0.5, c=colors, cmap='viridis',  marker='.')
    ax.plot([min(actual), max(actual)], [min(predicted), max(predicted)], color='r', linestyle='--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True)
    ax.axis('equal')
    ax.set_title(label)

def plot_theta_spread():
    plt.scatter(theta_1, theta_2, c=colors, cmap='viridis',  marker='.')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.grid(True)
    plt.axis('equal')
    plt.title(r'Spread for predicted values of $\theta_1$ and $\theta_2$')
    # Add text annotation
    spread_text = r'Range of $\theta_1 = {:.2f}$'.format(max(theta_1) - min(theta_1))
    plt.text(0.95, 0.95, spread_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

if __name__ == '__main__':
    
    np.random.seed(0)

    r = np.random.uniform(L1-L2, L1+L2, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    ang = np.linspace(0, 2*np.pi, 100) # To draw the workspace circles

    # Sorted in assending order
    r_sorted = np.sort(r)
    theta_sorted = np.sort(theta)

    ########################################################################################
    # FOR R SORTED

    X = r_sorted * np.cos(theta)
    Y = r_sorted * np.sin(theta)

    theta_1 = []
    theta_2 = []
    X_pred = []
    Y_pred = []

    for i in range(len(X)):
        input_example = np.array([[X[i], Y[i]]])
        input_tensor = torch.from_numpy(input_example).float()

        # Call the function to load and plot models
        theta1, theta2 = load_and_plot_models(MODEL_FOLDER, input_tensor)
        theta1 = np.array(theta1)
        theta2 = np.array(theta2)

        x, y = FK2R(theta1, theta2)

        # Convert to Python lists
        theta_1.append(theta1)
        theta_2.append(theta2)
        X_pred.append(x)
        Y_pred.append(y)

    C2 = np.cos(theta_2)

    # Ideal line
    x_ideal = [min(C2), max(C2)]
    c_ideal = L1**2 + L2**2
    m_ideal = 2*L1*L2
    y_ideal = [m_ideal * x + c_ideal for x in x_ideal]

    # Fit the model to the data
    params, covariance = curve_fit(linear_model, C2, (X**2 + Y**2))

    # Extract the parameters (slope and intercept)
    m_fit, c_fit = params

    # Generate points for the best-fit line
    C2_fit = np.linspace(min(C2), max(C2), 100)
    X2Y2_fit = linear_model(C2_fit, m_fit, c_fit)

    # Calculate R² score for X and Y
    r2X = r2_score(X, X_pred)
    r2Y = r2_score(Y, Y_pred)

    # PLOT
    plt.figure(figsize=(12, 8))

    # Plot the reachable workspace
    plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)  # Added this line
    plot_generated_points()

    # Plot the distribution for r and theta
    plot_distributions()

    # Plot X^2 + Y^2 = 2*L1*L2*cos(theta2) + L1^2 + L2^2
    plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    plot_2r_ik()

    # Plot parity plot for X
    plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), X, X_pred, r'Parity plot for X $\left( R^2 = {:.4f} \right)$'.format(r2X))

    # Plot parity plot for Y
    plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), Y, Y_pred, r'Parity plot for Y $\left( R^2 = {:.4f} \right)$'.format(r2Y))

    # Plot the spread of theta1 and theta2
    plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)
    plot_theta_spread()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'results_r_sorted.png'), dpi=600)

    ########################################################################################
    # FOR THETA SORTED

    X = r * np.cos(theta_sorted)
    Y = r * np.sin(theta_sorted)

    theta_1 = []
    theta_2 = []
    X_pred = []
    Y_pred = []

    for i in range(len(X)):
        input_example = np.array([[X[i], Y[i]]])
        input_tensor = torch.from_numpy(input_example).float()

        # Call the function to load and plot models
        theta1, theta2 = load_and_plot_models(MODEL_FOLDER, input_tensor)
        theta1 = np.array(theta1)
        theta2 = np.array(theta2)

        x, y = FK2R(theta1, theta2)

        # Convert to Python lists
        theta_1.append(theta1)
        theta_2.append(theta2)
        X_pred.append(x)
        Y_pred.append(y)

    C2 = np.cos(theta_2)

    # Ideal line
    x_ideal = [min(C2), max(C2)]
    c_ideal = L1**2 + L2**2
    m_ideal = 2*L1*L2
    y_ideal = [m_ideal * x + c_ideal for x in x_ideal]

    # Fit the model to the data
    params, covariance = curve_fit(linear_model, C2, (X**2 + Y**2))

    # Extract the parameters (slope and intercept)
    m_fit, c_fit = params

    # Generate points for the best-fit line
    C2_fit = np.linspace(min(C2), max(C2), 100)
    X2Y2_fit = linear_model(C2_fit, m_fit, c_fit)

    # Calculate R² score for X and Y
    r2X = r2_score(X, X_pred)
    r2Y = r2_score(Y, Y_pred)

    # PLOT
    plt.figure(figsize=(12, 8))

    # Plot the reachable workspace
    plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)  # Added this line
    plot_generated_points()

    # Plot the distribution for r and theta
    plot_distributions()

    # Plot X^2 + Y^2 = 2*L1*L2*cos(theta2) + L1^2 + L2^2
    plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    plot_2r_ik()

    # Plot parity plot for X
    plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), X, X_pred, r'Parity plot for X $\left( R^2 = {:.4f} \right)$'.format(r2X))

    # Plot parity plot for Y
    plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
    plot_parity_plot(plt.gca(), Y, Y_pred, r'Parity plot for Y $\left( R^2 = {:.4f} \right)$'.format(r2Y))

    # Plot the spread of theta1 and theta2
    plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)
    plot_theta_spread()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'results_theta_sorted.png'), dpi=600)

    # plt.show()