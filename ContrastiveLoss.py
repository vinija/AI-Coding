import torch
import torch.nn as nn
import torch.nn.functional as F


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    This loss function is used in the context of a siamese network.
    It helps the network learn to distinguish between pairs of similar and dissimilar items.

    Args:
    margin (float): Margin for contrastive loss. It defines the baseline for separating positive and negative pairs.
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Forward pass for the contrastive loss calculation.

        Args:
        output1 (torch.Tensor): Output from one of the twin networks.
        output2 (torch.Tensor): Output from the other twin network.
        label (torch.Tensor): Labels indicating if the pair is similar (1) or dissimilar (0).

        Returns:
        torch.Tensor: Computed contrastive loss.
        """
        # Calculate the Euclidean distance between the two outputs
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Calculate contrastive loss
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive



