# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

# PyTorch Imports
import torch                                                  # PyTorch
import torch.nn as nn                                         # Call neural network module
import torch.nn.init as init                                  # Initializing the model weights
import torch.optim as optim                                   # For optimization
from torchinfo import summary                                 # To get the summary of the neural network
from torch.utils.data import DataLoader, TensorDataset        #

# Other imports
import os
import sys
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
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate: Default: 0.01')
#
parser.add_argument('--results_suffix', type=str, default='#######', help='Suffix for results folder at each run')
args = parser.parse_args()


class inv2r_nn(nn.Module):
    def __init__(self):
        super(inv2r_nn, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
        )

    # Apply He initialization to the layers
        for layer in self.main:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.zeros_(layer.bias)

    def forward(self, x):
        output = self.main(x)
        return output


def save_metrics(metrics, filename='metrics.txt'):
    with open(filename, 'w') as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")


if __name__ == '__main__':

    CSD = os.path.dirname(__file__)
    BDR = os.path.dirname(CSD)

    DATA_FOLDER = os.path.join(BDR, "data")
    RESULTS_FOLDER = os.path.join(BDR, "_results", f"ik_nn_{args.results_suffix}")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    PATH_TO_CKPT = os.path.join(BDR, "models")

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    Print_Model_Summary = False

    # Preparing the dataset
    data = np.load(os.path.join(DATA_FOLDER, "DataN200.npy"))

    X = data[:, 2:4]
    y = data[:, 0:2]

    tr_ratio = 0.70
    val_ratio = 0.15
    ts_ratio = 1 - tr_ratio - val_ratio

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=ts_ratio, shuffle=True, random_state=143)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio / (1 - ts_ratio), shuffle=True, random_state=143)

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
    INV_MODEL1 = inv2r_nn().to(DEVICE)
    input_shape = (7, 2)
    summary(INV_MODEL1, input_size=input_shape, col_names=[
            'input_size', 'output_size']) if Print_Model_Summary else None

    # Loss function and optimizer
    mse_loss = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(INV_MODEL1.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Prepare dictionary to store metrics
    metrics = {'train_loss': [], 'val_loss': [], 'test_loss': []}

    num_tr_batch = len(train_dl)
    best_val_loss = float('inf')
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        tq_1 = f"[Epoch: {epoch+1:>3d}/{NUM_EPOCHS:<3d}]"
        INV_MODEL1.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            # Normalize the inputs if required! Here it is not required.

            # clear the gradients
            optimizer.zero_grad()

            # Forward Pass - computre the model output
            yhat = INV_MODEL1(inputs)

            # Loss calculation
            loss = mse_loss(yhat, targets)

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
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                yhat = INV_MODEL1(X)
                loss = mse_loss(yhat, y)
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
    predictions = []
    actuals = []

    INV_MODEL1.eval()
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            yhat = INV_MODEL1(X)
            predictions.append(yhat.cpu())
            actuals.append(y.cpu())

    # Convert lists to tensors
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)

    predictions = predictions.numpy()
    actuals = actuals.numpy()

    # Calculate R² score for each property
    r2_scores = [r2_score(actuals[:, i], predictions[:, i]) for i in range(actuals.shape[1])]
    for i, score in enumerate(r2_scores):
        print(f'R² score for theta_{i + 1}: {score:.4f}')

    num_targets = 2
    target_names = ['theta_1', 'theta_2']

    for i in range(num_targets):
        plt.figure(figsize=(8, 8))
        plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5, label='Test Points')

        # Adding a line y=x for reference
        min_val, max_val = min(actuals[:, i].min(), predictions[:, i].min()), max(
            actuals[:, i].max(), predictions[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Parity Plot for {target_names[i]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_FOLDER, f'parity_plot_{target_names[i]}.png'), dpi=300)

    # mae = [mean_absolute_error(actuals[:, i], predictions[:, i]) for i in range(num_targets)]
    # rmse = [np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i])) for i in range(num_targets)]
