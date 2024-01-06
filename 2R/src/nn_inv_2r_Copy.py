# https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
# https://github.com/mmc-group/inverse-designed-spinodoids

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
import time
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
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate: Default: 0.2')
#
parser.add_argument('--results_suffix', type=str, default='#######', help='Suffix for results folder at each run')

args = parser.parse_args()


def fk(theta_1, theta_2):
    L1 = 5
    L2 = 3
    X = L1 * torch.cos(theta_1) + L2 * torch.cos(theta_1 + theta_2)
    Y = L1 * torch.sin(theta_1) + L2 * torch.sin(theta_1 + theta_2)
    return X, Y


class Normalization:
    def __init__(self, data):
        # Assuming data is a torch.Tensor
        self.min = torch.min(data, dim=0)[0]
        self.max = torch.max(data, dim=0)[0]

    def normalize(self, data):
        # Element-wise normalization
        return (data - self.min) / (self.max - self.min)

    def unnormalize(self, data):
        # Element-wise denormalization
        return data * (self.max - self.min) + self.min


class inv2r_nn(nn.Module):
    def __init__(self):
        super(inv2r_nn, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.Softplus(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        output = self.main(x)
        return output

    # def weights_init_uniform_rule(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             n = m.in_features
    #             y = 1.0 / np.sqrt(n)
    #             m.weight.data.uniform_(-y, y)
    #             m.bias.data.fill_(0)

    def weights_init_uniform_rule(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def save_metrics(metrics, filename='metrics.txt'):
    with open(filename, 'w') as f:
        for key, values in metrics.items():
            f.write(f"{key}: {values}\n")


if __name__ == '__main__':

    start_time = time.time()

    CSD = os.path.dirname(__file__)
    BDR = os.path.dirname(CSD)

    DATA_FOLDER = os.path.join(BDR, "data")
    RESULTS_FOLDER = os.path.join(BDR, "_results", f"ik_nn_{args.results_suffix}")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    PATH_TO_CKPT = RESULTS_FOLDER

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCH_ZETA = args.zeta_factor * NUM_EPOCHS
    ZETA = args.zeta
    Print_Model_Summary = False

    # Preparing the dataset
    data = np.load(os.path.join(DATA_FOLDER, "DataN200.npy"))

    # Input and output for inverse problem
    X = data[:, 2:4]  # Position
    y = data[:, 0:2]  # Thetas

    # # Input and output for forward problem
    # X = data[:, 0:2]  # Thetas
    # y = data[:, 2:4]  # Position

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

    # Creating the normalizer from train dataset
    X_normalizer = Normalization(X_train)

    # Normalizing the inputs
    X_train = X_normalizer.normalize(X_train)
    X_val = X_normalizer.normalize(X_val)
    X_test = X_normalizer.normalize(X_test)

    tr_opt = {'batch_size': BATCH_SIZE, 'shuffle': True, 'pin_memory': True}
    test_val_opt = {'batch_size': BATCH_SIZE, 'shuffle': False, 'pin_memory': True}

    # Preparing PyTorch Dataloader
    train_dl = DataLoader(TensorDataset(X_train, y_train), **tr_opt)
    val_dl = DataLoader(TensorDataset(X_val, y_val), **test_val_opt)
    test_dl = DataLoader(TensorDataset(X_test, y_test), **test_val_opt)

    # Model instance and print summary
    INV_MODEL1 = inv2r_nn().to(DEVICE)
    INV_MODEL1.weights_init_uniform_rule()
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
    predictions = []
    actuals = []

    INV_MODEL1.eval()
    with torch.no_grad():
        for pos_test, theta_test in test_dl:
            pos_test, theta_test = pos_test.to(DEVICE), theta_test.to(DEVICE)
            theta_test_pred = INV_MODEL1(pos_test)
            predictions.append(theta_test_pred.cpu())
            actuals.append(theta_test.cpu())

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

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds")

    # mae = [mean_absolute_error(actuals[:, i], predictions[:, i]) for i in range(num_targets)]
    # rmse = [np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i])) for i in range(num_targets)]
