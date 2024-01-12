# Torch imports
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Other imports
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Modules
import config
from model import fk4r_freud
from config import parse_arg
from utils import (save_hyperparameters_md,
                   calculate_r2_scores,
                   generate_parity_plots,
                   save_metrics,
                   )


def save_current_script_folder(results_folder):
    current_script_dir = os.path.dirname(__file__)
    destination_dir_path = os.path.join(results_folder, os.path.basename(current_script_dir))

    # Check if the destination directory exists, and remove it if it does
    if os.path.exists(destination_dir_path):
        shutil.rmtree(destination_dir_path)

    shutil.copytree(current_script_dir, destination_dir_path)


if __name__ == '__main__':
    args = parse_arg()
    CSD = os.path.dirname(__file__)
    BDR = os.path.dirname(CSD)
    RESULTS_DIR = os.path.join(BDR, "_results", f"{args.results_suffix}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save hyperparameters in a markdown file and current script folder as well
    md_file_path = os.path.join(RESULTS_DIR, "hyperparameters.md")
    save_hyperparameters_md(md_file_path)
    save_current_script_folder(RESULTS_DIR)

    data_file = "data_2000"

    DATA_FOLDER = os.path.join(BDR, "data", f"{data_file}")
    PATH_TO_CKPT = RESULTS_DIR

    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Preparing the dataset
    data_train_temp = np.load(os.path.join(DATA_FOLDER, "train_data.npy"))
    data_test = np.load(os.path.join(DATA_FOLDER, "test_data.npy"))

    tr_ratio = 0.70
    val_ratio = 0.15
    ts_ratio = 1 - tr_ratio - val_ratio

    data_train, data_val = train_test_split(data_train_temp, test_size=val_ratio / (1 - ts_ratio),
                                            shuffle=True, random_state=143)

    # X = theta1, y = phi1
    X_train = data_train[:, 0]
    y_train = data_train[:, 1]
    X_val = data_val[:, 0]
    y_val = data_val[:, 1]
    X_test = data_test[:, 0]
    y_test = data_test[:, 1]

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

    # Model instance
    FREUD_MODEL = fk4r_freud(input_dim=config.input_dim_freud, hidden_dims=args.hidden_dims,
                             output_dim=config.output_dim_freud).to(config.DEVICE)
    FREUD_MODEL.apply(FREUD_MODEL.weights_init_uniform_rule)

    # Loss function and optimizer
    mse_loss = torch.nn.MSELoss()  # Mean Squared Error Loss
    # optimizer = optim.SGD(FREUD_MODEL.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optimizer = torch.optim.Adam(FREUD_MODEL.parameters(), lr=LEARNING_RATE)

    # Prepare dictionary to store metrics
    metrics = {'train_loss': [], 'val_loss': [], 'test_loss': []}

    num_tr_batch = len(train_dl)
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        tq_1 = f"[Epoch: {epoch+1:>3d}/{NUM_EPOCHS:<3d}]"
        FREUD_MODEL.train()
        train_loss = 0.0

        for batch_idx, (theta, phi) in enumerate(train_dl):
            theta, phi = theta.unsqueeze(1), phi.unsqueeze(1)
            theta, phi = theta.to(config.DEVICE), phi.to(config.DEVICE)

            # clear the gradients
            optimizer.zero_grad()

            # Forward Pass - compute the model output
            phi_pred = FREUD_MODEL(theta)

            # Loss calculation
            loss = mse_loss(phi_pred, phi)

            # backpass gradients
            loss.backward()

            # update the model weights
            optimizer.step()

            train_loss += loss.item()
            print(tq_1 + f"[Iter: {batch_idx+1:>3d}/{num_tr_batch:<3d}]", end="\r")
        # end for batch_idx

        # Validate the val_dl at each epoch
        FREUD_MODEL.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_val, phi_val in val_dl:
                theta_val, phi_val = theta_val.unsqueeze(1), phi_val.unsqueeze(1)
                theta_val, phi_val = theta_val.to(config.DEVICE), phi_val.to(config.DEVICE)
                phi_pred = FREUD_MODEL(theta_val)
                # pos_pred = fk(theta_pred[0], theta_pred[1]).to(DEVICE)
                loss = mse_loss(phi_pred, phi_val)
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
                'net_state_dict': FREUD_MODEL.state_dict(),
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
        plt.savefig(os.path.join(RESULTS_DIR, 'training_val_loss.png'))
        plt.close()

        save_metrics(metrics, os.path.join(RESULTS_DIR, "metrics.txt"))
    # End of epoch

    # Load the best model for testing
    checkpoint = torch.load(ckpt_save_file)
    FREUD_MODEL.load_state_dict(checkpoint['net_state_dict'])

    # Prepare to store predictions and actual values
    FREUD_MODEL.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for theta_test, phi_test in test_dl:
            theta_test, phi_test = theta_test.unsqueeze(1), phi_test.unsqueeze(1)
            theta_test, phi_test = theta_test.to(config.DEVICE), phi_test.to(config.DEVICE)
            phi_test_pred = FREUD_MODEL(theta_test)

            # Collect predictions and actual values
            predictions.append(phi_test_pred.cpu())
            actuals.append(phi_test.cpu())

    # Convert lists to tensors and then to numpy arrays
    predictions, actuals = torch.cat(predictions, dim=0).numpy(), torch.cat(actuals, dim=0).numpy()

    # Calculate and print R² scores
    calculate_r2_scores(actuals, predictions, ['phi_1'], RESULTS_DIR)

    # Generate and save parity plots
    generate_parity_plots(actuals, predictions, ['phi_1'], RESULTS_DIR)
