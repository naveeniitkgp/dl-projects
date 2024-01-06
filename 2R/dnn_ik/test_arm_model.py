import torch
import numpy as np
from train_arm_model import inv2r_nn

import os


def load_model(model_path):
    model = inv2r_nn().to('cpu')  # Assuming the model is to be run on CPU
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['net_state_dict'])
    model.eval()
    return model


def predict(model, input_data):
    with torch.no_grad():
        prediction = model(input_data)
        return prediction


if __name__ == '__main__':
    CSD = os.path.dirname(__file__)
    BDR = os.path.dirname(CSD)

    # Path to your trained model checkpoint
    model_path = os.path.join(BDR, "_results", "dnn_ik_best_1", "best_model_epoch_190.pth")

    # Load the model
    model = load_model(model_path)

    theta_1 = 2
    theta_2 = -1
    delta = 0.0
    x_des = np.cos(theta_1) + np.cos(theta_2)
    y_des = np.sin(theta_1) + np.sin(theta_2)

    # Example input - replace with your actual input
    print("Enter the value of end effector position")
    input_example = np.array([[x_des, y_des, (theta_1 + delta), (theta_2 + delta)]])  # Shape: (1, 4)
    input_tensor = torch.from_numpy(input_example).float()

    # Get prediction
    output = predict(model, input_tensor)

    # Print prediction
    print("Predicted Output:", output.numpy())
