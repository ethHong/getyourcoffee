import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GeneralizedSigmoid(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # w^T * x + b

        # Generalized sigmoid parameters: L, k, x0, offset (all learnable)
        self.L = nn.Parameter(torch.tensor(15.0))  # Max value
        self.k = nn.Parameter(torch.tensor(1.0))  # Slope
        self.x0 = nn.Parameter(torch.tensor(0.0))  # Midpoint
        self.offset = nn.Parameter(torch.tensor(0.0))  # Minimum value

    def forward(self, x):
        z = self.linear(x)  # Linear transformation
        sigmoid_output = self.L / (1 + torch.exp(-self.k * (z - self.x0))) + self.offset
        return sigmoid_output
