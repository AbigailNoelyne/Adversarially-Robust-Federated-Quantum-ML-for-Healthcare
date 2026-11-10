import torch
from torch import nn

class CirrhosisPredictor(nn.Module):
    """
    A neural network model for predicting cirrhosis stages.

    This model uses a series of fully connected layers with ReLU activations
    and dropout for regularization. The final layer uses softmax activation
    for multi-class classification.

    Args:
        input_dim (int): The number of input features.
        output_dim (int, optional): The number of output classes. Defaults to 3.

    Attributes:
        fc (nn.Sequential): The sequence of fully connected layers that make up the model.
    """
    def __init__(self, input_dim, output_dim=3):
        super(CirrhosisPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output predictions after passing through the model.
        """
        return self.fc(x)