import numpy as np
from solution_tutor import NeuralNetworkModel

"""
DO NOT CHANGE CODE IN THIS FILE
"""


def init_toy_data(num_inputs=5, input_size=4):
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y


def init_toy_model(input_size, hidden_size, num_classes):
    np.random.seed(0)
    return NeuralNetworkModel(input_size, hidden_size, num_classes, std=1e-1)