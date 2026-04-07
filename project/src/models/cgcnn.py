"""Crystal Graph Convolutional Neural Network (CGCNN).

Reference: Xie & Grossman, PRL 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CGCNNConv(nn.Module):
    """Single graph convolution layer for crystal graphs."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus = nn.Softplus()

    def forward(
        self,
        atom_fea: torch.Tensor,   # (N_total, atom_fea_len)
        nbr_fea: torch.Tensor,    # (N_total, M, nbr_fea_len)
        nbr_idx: torch.Tensor,    # (N_total, M)
    ) -> torch.Tensor:
        N, M = nbr_idx.shape

        # Gather neighbor atom features
        atom_nbr_fea = atom_fea[nbr_idx]  # (N, M, atom_fea_len)

        # Expand center atom features
        atom_fea_expand = atom_fea.unsqueeze(1).expand(-1, M, -1)  # (N, M, atom_fea_len)

        # Concatenate: [center, neighbor, bond]
        total_fea = torch.cat([atom_fea_expand, atom_nbr_fea, nbr_fea], dim=2)
        # (N, M, 2*atom_fea_len + nbr_fea_len)

        total_fea = self.fc_full(total_fea)  # (N, M, 2*atom_fea_len)
        total_fea = self.bn1(total_fea.view(-1, total_fea.shape[-1])).view(N, M, -1)

        # Split into filter and core
        filter_fea, core_fea = total_fea.chunk(2, dim=2)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.softplus(core_fea)

        # Aggregate over neighbors
        nbr_msg = (filter_fea * core_fea).sum(dim=1)  # (N, atom_fea_len)
        nbr_msg = self.bn2(nbr_msg)

        # Residual connection
        return atom_fea + self.softplus(nbr_msg)


class CGCNN(nn.Module):
    """Full CGCNN model: embedding -> convolutions -> pooling -> MLP."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        atom_fea_len: int = 64,
        nbr_fea_len: int = 41,  # GaussianDistance default: (8.0 - 0.0) / 0.2 + 1 = 41
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.convs = nn.ModuleList(
            [CGCNNConv(atom_fea_len, nbr_fea_len) for _ in range(n_conv)]
        )

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        hidden_layers = []
        for _ in range(n_h - 1):
            hidden_layers.append(nn.Linear(h_fea_len, h_fea_len))
            hidden_layers.append(nn.Softplus())
        self.hidden = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()

        self.fc_out = nn.Linear(h_fea_len, output_dim)

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            atom_fea: (N_total, orig_atom_fea_len)
            nbr_fea:  (N_total, M, nbr_fea_len)
            nbr_idx:  (N_total, M)
            crystal_atom_idx: (N_total,) - maps atom to crystal in batch

        Returns:
            (batch_size, output_dim)
        """
        # Embed
        atom_fea = self.embedding(atom_fea)

        # Graph convolutions
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)

        # Pool: mean over atoms per crystal
        batch_size = crystal_atom_idx.max().item() + 1
        crys_fea = torch.zeros(batch_size, atom_fea.shape[1], device=atom_fea.device)
        count = torch.zeros(batch_size, 1, device=atom_fea.device)

        crys_fea.scatter_add_(0, crystal_atom_idx.unsqueeze(1).expand_as(atom_fea), atom_fea)
        count.scatter_add_(0, crystal_atom_idx.unsqueeze(1), torch.ones(atom_fea.shape[0], 1, device=atom_fea.device))
        crys_fea = crys_fea / count.clamp(min=1)

        # MLP head
        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        crys_fea = self.hidden(crys_fea)
        out = self.fc_out(crys_fea)
        return out

    def get_embedding(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return the crystal-level embedding before the final FC layer."""
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)

        batch_size = crystal_atom_idx.max().item() + 1
        crys_fea = torch.zeros(batch_size, atom_fea.shape[1], device=atom_fea.device)
        count = torch.zeros(batch_size, 1, device=atom_fea.device)
        crys_fea.scatter_add_(0, crystal_atom_idx.unsqueeze(1).expand_as(atom_fea), atom_fea)
        count.scatter_add_(0, crystal_atom_idx.unsqueeze(1), torch.ones(atom_fea.shape[0], 1, device=atom_fea.device))
        crys_fea = crys_fea / count.clamp(min=1)

        crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))
        crys_fea = self.hidden(crys_fea)
        return crys_fea
