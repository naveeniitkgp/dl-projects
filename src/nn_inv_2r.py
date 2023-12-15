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
parser.add_argument('--batch_size', type=float, default=32, help='Batch size: Default: 32')
parser.add_argument('--num_epochs', type=float, default=500, help='Number of Epochs: Default: 500')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate: Default: 0.01')
args = parser.parse_args()


class inv2r_nn(nn.Module):
    def __init__(self):
        super(inv2r_nn, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
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
    RESULTS_FOLDER = os.path.join(BDR, "_results")

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    Print_Model_Summary = False

    # Preparing the dataset
    data = np.load(os.path.join(DATA_FOLDER, "DataN200.npy"))

    X = data[:, 2:4]
    y = data[:, 0:2]

    tr_ratio = 0.80
    ts_ratio = 1 - tr_ratio

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts_ratio, shuffle=True, random_state=143)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    tr_opt = {'batch_size': BATCH_SIZE, 'shuffle': True, 'pin_memory': True}
    test_opt = {'batch_size': BATCH_SIZE, 'shuffle': False, 'pin_memory': True}

    # Preparing PyTorch Dataloader
    train_dl = DataLoader(TensorDataset(X_train, y_train), **tr_opt)
    test_dl = DataLoader(TensorDataset(X_test, y_test), **test_opt)

    # Model instance and print summary
    INV_MODEL1 = inv2r_nn().to(DEVICE)
    input_shape = (7, 2)
    summary(INV_MODEL1, input_size=input_shape, col_names=[
            'input_size', 'output_size']) if Print_Model_Summary else None

    # Loss function and optimizer
    mse_loss = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.SGD(INV_MODEL1.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Prepare dictionary to store metrics
    metrics = {'train_loss': [], 'test_loss': []}

    num_tr_batch = len(train_dl)
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

        # Validate the Test data at each epoch
        INV_MODEL1.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                yhat = INV_MODEL1(X)
                loss = mse_loss(yhat, y)
                test_loss += loss.item()

        train_loss /= len(train_dl)
        test_loss /= len(test_dl)
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)

        # Plotting Training and Test Loss
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['test_loss'], label='Test Loss')
        plt.yscale('log')
        plt.title('Training and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_FOLDER, 'training_test_loss.png'))

        save_metrics(metrics, os.path.join(RESULTS_FOLDER, "metrics.txt"))
    # End of epoch
