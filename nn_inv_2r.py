# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# PyTorch Imports
import torch                                                  # PyTorch
import torch.nn as nn                                         # Call neural network module
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
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate: Default: 0.01')
args = parser.parse_args()


class inv2r_nn(nn.Module):
    def __init__(self):
        super(inv2r_nn, self).__init__()
        self.linear = nn.Sequential(
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

    def forward(self, x):
        output = self.linear(x)
        return output


Print_Model_Summary = True
INV_MODEL1 = inv2r_nn().to(DEVICE)

input_shape = (7, 2)

summary(INV_MODEL1, input_size=input_shape, col_names=['input_size', 'output_size']) if Print_Model_Summary else None
