# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset

# Other imports
import os
import sys
import shutil
import timeit
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size: Default: 32')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of Epochs: Default: 500')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate: Default: 0.001')
parser.add_argument('--zeta_factor', type=float, default=0.2,
                    help='percentage of epochs till zeta is used: Default: 0.2')
parser.add_argument('--zeta', type=float, default=0.5, help='Zeta value: Default: 0.5')
#
parser.add_argument('--results_suffix', type=str, default='#######', help='Suffix for results folder at each run')
parser.add_argument('--hidden_dims', nargs='+', type=int,
                    default=[100, 150], help='List of the number of neurons in each hidden layer. Default: 100 150')

args = parser.parse_args()


def fk(theta_1, theta_2):
    L1 = 5
    L2 = 3
    X = L1 * torch.cos(theta_1) + L2 * torch.cos(theta_1 + theta_2)
    Y = L1 * torch.sin(theta_1) + L2 * torch.sin(theta_1 + theta_2)
    return X, Y


class inv2r_nn(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[100, 150, 200, 150, 100], output_dim=2):
        super(inv2r_nn, self).__init__()

        # Initialize layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dims[0], bias=False))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Softplus())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Softplus())

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine all layers into a Sequential module
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        output = self.main(x)
        return output

    def weights_init_uniform_rule(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def save_metrics(metrics, filename='metrics.txt'):
    with open(filename, 'w') as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")


def save_hyperparameters_md(args, file_path):
    # Convert hidden_dims list to a string for display
    hidden_dims_str = ', '.join(map(str, args.hidden_dims))

    # Create a Markdown table with hyperparameters and their values
    markdown_content = f"""
| Hyperparameters       | Values          |
|:---------------------:|:---------------:|
| NUM_EPOCHS            | {args.num_epochs} |
| BATCH_SIZE            | {args.batch_size} |
| LEARNING_RATE         | {args.lr}        |
| EPOCH_ZETA            | {args.zeta_factor} |
| ZETA                  | {args.zeta}      |
| RESULT_FOLDER_SUFFIX  | {args.results_suffix} |
| HIDDEN_LAYER_DIMS     | [{hidden_dims_str}] |
"""

    # Write the Markdown content to the specified file
    with open(file_path, 'w') as file:
        file.write(markdown_content)


def save_current_script(results_folder):
    current_script_path = __file__
    destination_script_path = os.path.join(results_folder, os.path.basename(current_script_path))
    shutil.copy(current_script_path, destination_script_path)


def calculate_r2_scores(actuals, predictions, target_names):
    for i in range(len(target_names)):
        score = r2_score(actuals[:, i], predictions[:, i])
        print(f'R² score for {target_names[i]}: {score:.4f}')


def generate_parity_plots(actuals, predictions, target_names, results_folder):
    for i, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5, label='Test Points')
        min_val, max_val = min(actuals[:, i].min(), predictions[:, i].min()), max(
            actuals[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Parity Plot for {target_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_folder, f'parity_plot_{target_name}.png'), dpi=300)
        plt.close()


if __name__ == '__main__':

    CSD = os.path.dirname(__file__)
    BDR = os.path.dirname(CSD)

    data_name = "N100_theta12_+-180"

    DATA_FOLDER = os.path.join(BDR, "data", f"{data_name}")
    RESULTS_FOLDER = os.path.join(BDR, "_results", f"ik_nn_{args.results_suffix}")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    PATH_TO_CKPT = RESULTS_FOLDER

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCH_ZETA = args.zeta_factor * NUM_EPOCHS
    ZETA = args.zeta
    Print_Model_Summary = False

    # Parameters for the model
    input_dim = 2
    output_dim = 2
    hidden_dims = args.hidden_dims

    # Save hyperparameters in a markdown file
    md_file_path = os.path.join(RESULTS_FOLDER, "hyperparameters.md")
    save_hyperparameters_md(args, md_file_path)

    # Saving the main Python script (the one you're currently running) to the results folder
    save_current_script(RESULTS_FOLDER)

    # Preparing the dataset
    data_train_temp = np.load(os.path.join(DATA_FOLDER, "train_data.npy"))
    data_test = np.load(os.path.join(DATA_FOLDER, "test_data.npy"))

    tr_ratio = 0.70
    val_ratio = 0.15
    ts_ratio = 1 - tr_ratio - val_ratio

    data_train, data_val = train_test_split(data_train_temp, test_size=val_ratio / (1 - ts_ratio),
                                            shuffle=True, random_state=143)

    # X = Position, y = Thetas
    X_train = data_train[:, 2:4]
    y_train = data_train[:, 0:2]
    X_val = data_val[:, 2:4]
    y_val = data_val[:, 0:2]
    X_test = data_test[:, 2:4]
    y_test = data_test[:, 0:2]

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    y_test = torch.from_numpy(y_test).float()

    tr_opt = {'batch_size': BATCH_SIZE, 'shuffle': True, 'pin_memory': True}
    test_val_opt = {'batch_size': BATCH_SIZE, 'shuffle': False, 'pin_memory': True}

    # Preparing PyTorch Dataloader
    train_dl = DataLoader(TensorDataset(X_train, y_train), **tr_opt)
    val_dl = DataLoader(TensorDataset(X_val, y_val), **test_val_opt)
    test_dl = DataLoader(TensorDataset(X_test, y_test), **test_val_opt)

    # Model instance and print summary
    INV_MODEL1 = inv2r_nn(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(DEVICE)
    # INV_MODEL1.weights_init_uniform_rule()
    INV_MODEL1.apply(INV_MODEL1.weights_init_uniform_rule)
    # FWD_MODEL = fk().to(DEVICE)
    input_shape = (7, 2)
    summary(INV_MODEL1, input_size=input_shape, col_names=[
            'input_size', 'output_size']) if Print_Model_Summary else None

    # Loss function and optimizer
    inv_loss = torch.nn.MSELoss()  # Mean Squared Error Loss
    fwd_loss = torch.nn.MSELoss()  # Mean Squared Error Loss
    # optimizer = optim.SGD(INV_MODEL1.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optimizer = torch.optim.Adam(INV_MODEL1.parameters(), lr=LEARNING_RATE)

    # Prepare dictionary to store metrics
    metrics = {'train_loss': [], 'val_loss': [], 'test_loss': []}

    num_tr_batch = len(train_dl)
    best_val_loss = float('inf')
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        tq_1 = f"[Epoch: {epoch+1:>3d}/{NUM_EPOCHS:<3d}]"
        INV_MODEL1.train()
        train_loss = 0.0

        if (epoch > EPOCH_ZETA):
            ZETA = 0.0

        for batch_idx, (pos, theta) in enumerate(train_dl):
            pos, theta = pos.to(DEVICE), theta.to(DEVICE)
            # Normalize the pos if required! Here it is not required.

            # clear the gradients
            optimizer.zero_grad()

            # Forward Pass - compute the model output
            theta_pred = INV_MODEL1(pos)

            # calculating the position using theta_pred
            # pos_pred = fk(theta_pred[0], theta_pred[1]).to(DEVICE)

            pos_pred_x, pos_pred_y = fk(theta_pred[:, 0], theta_pred[:, 1])
            pos_pred = torch.stack([pos_pred_x, pos_pred_y], dim=1)

            # Loss calculation
            loss = inv_loss(pos_pred, pos) + ZETA * fwd_loss(theta_pred, theta)

            # backpass gradients
            loss.backward()

            # update the model weights
            optimizer.step()

            train_loss += loss.item()
            print(tq_1 + f"[Iter: {batch_idx+1:>3d}/{num_tr_batch:<3d}]", end="\r")
        # end for batch_idx

        # Validate the val_dl at each epoch
        INV_MODEL1.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pos_val, theta_val in val_dl:
                pos_val, theta_val = pos_val.to(DEVICE), theta_val.to(DEVICE)
                theta_pred = INV_MODEL1(pos_val)
                # pos_pred = fk(theta_pred[0], theta_pred[1]).to(DEVICE)
                pos_pred_x, pos_pred_y = fk(theta_pred[:, 0], theta_pred[:, 1])
                pos_pred = torch.stack([pos_pred_x, pos_pred_y], dim=1)
                loss = inv_loss(pos_pred, pos_val) + ZETA * fwd_loss(theta_pred, theta_val)
                val_loss += loss.item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)

        # Save the model using val_loss
        if val_loss < best_val_loss:
            ckpt_save_file = PATH_TO_CKPT + "/ckpt_in_train/best_model_epoch_{}.pth".format(epoch + 1)
            os.makedirs(os.path.dirname(ckpt_save_file), exist_ok=True)
            torch.save({
                'net_state_dict': INV_MODEL1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_save_file)
            best_val_loss = val_loss

        # Plotting Training and Test Loss
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['val_loss'], label='Val Loss')
        plt.yscale('log')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_FOLDER, 'training_val_loss.png'))
        plt.close()

        save_metrics(metrics, os.path.join(RESULTS_FOLDER, "metrics.txt"))
    # End of epoch

    # Load the best model for testing
    checkpoint = torch.load(ckpt_save_file)
    INV_MODEL1.load_state_dict(checkpoint['net_state_dict'])

    # Prepare to store predictions and actual values
    INV_MODEL1.eval()
    predictions, actuals = [], []
    pos_predictions, pos_actuals = [], []

    with torch.no_grad():
        for pos_test, theta_test in test_dl:
            pos_test, theta_test = pos_test.to(DEVICE), theta_test.to(DEVICE)
            theta_test_pred = INV_MODEL1(pos_test)
            pos_pred_x, pos_pred_y = fk(theta_test_pred[:, 0], theta_test_pred[:, 1])
            pos_pred_test = torch.stack([pos_pred_x, pos_pred_y], dim=1)

            # Collect predictions and actual values
            predictions.append(theta_test_pred.cpu())
            actuals.append(theta_test.cpu())
            pos_predictions.append(pos_pred_test.cpu())
            pos_actuals.append(pos_test.cpu())

    # Convert lists to tensors and then to numpy arrays
    predictions, actuals = torch.cat(predictions, dim=0).numpy(), torch.cat(actuals, dim=0).numpy()
    pos_predictions, pos_actuals = torch.cat(pos_predictions, dim=0).numpy(), torch.cat(pos_actuals, dim=0).numpy()

    # Calculate and print R² scores
    calculate_r2_scores(actuals, predictions, ['theta_1', 'theta_2'])
    calculate_r2_scores(pos_actuals, pos_predictions, ['X', 'Y'])

    # Generate and save parity plots
    generate_parity_plots(actuals, predictions, ['theta_1', 'theta_2'], RESULTS_FOLDER)
    generate_parity_plots(pos_actuals, pos_predictions, ['X', 'Y'], RESULTS_FOLDER)
