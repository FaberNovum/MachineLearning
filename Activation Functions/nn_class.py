import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_function):
        """
        Initializes a neural network object.

        Args:
            input_size (int): The number of input features to the neural network.
            hidden_size (int): The number of neurons in the hidden layer.
            num_classes (int): The number of classes in the output layer.
            activation_function (torch.nn.Module): The activation function to be used in the network.

        Returns:
            None
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        x = self.layer3(x)
        return x
