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

import config
from config import parse_arg


class fk4r_freud(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[100, 150, 200, 150, 100], output_dim=1):
        super(fk4r_freud, self).__init__()

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


class fk4r_coupler(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[100, 150, 200, 150, 100], output_dim=2):
        super(fk4r_coupler, self).__init__()

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


if __name__ == '__main__':

    args = parse_arg()

    # Model instance and print summary
    FREUD_MODEL = fk4r_freud(input_dim=config.input_dim_freud, hidden_dims=args.hidden_dims,
                             output_dim=config.output_dim_freud).to(config.DEVICE)
    FREUD_MODEL.apply(FREUD_MODEL.weights_init_uniform_rule)
    input_shape = (7, config.input_dim_freud)
    summary(FREUD_MODEL, input_size=input_shape,
            col_names=['input_size', 'output_size']) if args.print_summary else None

    # Model instance and print summary
    COUPLER_MODEL = fk4r_coupler(input_dim=config.input_dim_coupler, hidden_dims=args.hidden_dims,
                                 output_dim=config.output_dim_coupler).to(config.DEVICE)
    COUPLER_MODEL.apply(COUPLER_MODEL.weights_init_uniform_rule)
    input_shape = (7, config.input_dim_coupler)
    summary(COUPLER_MODEL, input_size=input_shape,
            col_names=['input_size', 'output_size']) if args.print_summary else None
