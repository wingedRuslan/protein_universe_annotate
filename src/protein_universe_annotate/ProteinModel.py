import numpy as np
import torch
import torch.nn.functional as F


class SoftmaxRegression(torch.nn.Module):
    """ Softmax regression model for multi-class classification """

    def __init__(self, num_features, num_classes):
        """
        Args:
            num_features  (int): Number of input features
            num_classes (int): Number of output classes
        """
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the SoftmaxRegression model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            logits (torch.Tensor): Logits before applying softmax.
            probas (torch.Tensor): Probability distribution over the classes.
        """
        # Pass x through a parameterized linear transformation
        logits = self.linear(x)

        # Pass logits through softmax to generate a probability distribution vector over the classes
        probas = F.softmax(logits, dim=1)
        return logits, probas


