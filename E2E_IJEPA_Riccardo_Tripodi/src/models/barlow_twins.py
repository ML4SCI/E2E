from typing import Dict

import torch
from torch import nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """Barlow Twins model."""

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        lamb: float,
        batch_size: int,
    ):
        """Initializes the Barlow Twins model.

        Args:
            encoder (nn.Module): encoder to use.
            projector (nn.Module): projector to use.
            lamb (float): lambda parameter.
            batch_size (int): batch size.
            device (str): device to use.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.batch_size = batch_size
        self.lamb = lamb

        # Find out encoder len
        self.bn = nn.BatchNorm1d(projector[-1].out_features, affine=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass the Barlow Twins model."""
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # Empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.batch_size)

        # Sum the on-diagonal elements
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # Sum the off-diagonal elements
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamb * off_diag
        loss_normalized = (1 - self.lamb) / c.shape[0] * on_diag + self.lamb / (
            c.shape[0] * c.shape[0] - c.shape[0]
        ) * off_diag

        return {
            "loss": loss,
            "loss_norm": loss_normalized,
            "on_diag": on_diag,
            "off_diag": off_diag,
        }
