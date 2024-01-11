import torch
import torch.nn as nn

# Define a simple convolutional layer in PyTorch
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # Define a convolutional layer
        # nn.Conv2d parameters:
        # in_channels (int): Number of channels in the input image
        # out_channels (int): Number of channels produced by the convolution
        # kernel_size (int or tuple): Size of the convolving kernel
        # stride (int or tuple, optional): Stride of the convolution. Default: 1
        # padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Apply the convolutional layer to the input
        x = self.conv1(x)
        return x

def test_conv():

    # Example usage of the SimpleConvNet
    # Create an instance of the SimpleConvNet
    net = SimpleConvNet()

    # Create a dummy input tensor
    # For example, a single-channel image (e.g., grayscale) of size 64x64
    input_tensor = torch.randn(1, 1, 64, 64)  # Shape: [batch_size, channels, height, width]

    # Forward pass through the convolutional layer
    output_tensor = net(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    # The output shape will have 3 channels (as defined in out_channels of conv1),
    # and the spatial dimensions will remain the same (64x64) due to padding=1 and kernel_size=3.
