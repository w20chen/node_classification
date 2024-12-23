"""SIGN: Scalable Inception Graph Neural Networks (SIGN)

Paper link: https://arxiv.org/pdf/2004.11198
"""

import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """L-layer fully connected network"""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0):
        super().__init__()
        self.fc = nn.ModuleList()
        if num_layers == 1:
            self.fc.append(nn.Linear(in_dim, out_dim, bias=False))
        else:
            self.fc.append(nn.Linear(in_dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.fc.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.fc.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: tensor(N, d_in)
        :return: tensor(N, d_out)
        """
        for i, fc in enumerate(self.fc):
            x = fc(x)
            if i < len(self.fc) - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_hops, num_layers, dropout=0.0):
        """SIGN model

        :param in_dim: int Input feature dimension
        :param hidden_dim: int Hidden feature dimension
        :param out_dim: int Output feature dimension
        :param num_hops: int Number of hops r
        :param num_layers: int Number of layers in the fully connected network
        :param dropout: float, optional Dropout probability, default is 0
        """
        super().__init__()
        self.inception_ffs = nn.ModuleList([
            FeedForwardNet(in_dim, hidden_dim, hidden_dim, num_layers, dropout)  # Θ_i in equation (4)
            for _ in range(num_hops + 1)
        ])
        self.project = FeedForwardNet(
            (num_hops + 1) * hidden_dim, hidden_dim, out_dim, num_layers, dropout
        )  # Ω in equation (4)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats):
        """
        :param feats: List[tensor(N, d_in)] Aggregated features of neighbors for each hop, length is r+1
        :return: tensor(N, d_out) Output vertex features
        """
        # (N, (r+1)*d_hid)
        h = torch.cat([ff(feat) for ff, feat in zip(self.inception_ffs, feats)], dim=-1)
        out = self.project(self.dropout(self.prelu(h)))  # (N, d_out)
        return out
