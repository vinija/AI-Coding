# Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutNet(nn.Module):
    """
    Neural network with a dropout layer.
    Dropout is a regularization technique where randomly selected neurons are ignored during training.
    This helps prevent overfitting.

    Args:
    input_size (int): Size of the input layer.
    hidden_size (int): Size of the hidden layer.
    output_size (int): Size of the output layer.
    dropout_p (float): Probability of an element to be zeroed.
    """

    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(DropoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after the first linear layer
        x = self.dropout(x)  # Apply dropout to the activations of the first layer
        x = self.fc2(x)  # Apply the second linear layer
        return x

def test_dropout():
    # Example usage of DropoutNet
    input_size = 20
    hidden_size = 50
    output_size = 10
    dropout_p = 0.5

    net = DropoutNet(input_size, hidden_size, output_size, dropout_p)
    input_tensor = torch.randn(1, input_size)
    output_tensor = net(input_tensor)

    print("Output tensor shape:", output_tensor.shape)