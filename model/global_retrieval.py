from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalSourceRetrievalHead(nn.Module):

    def __init__(self, input_dim: int, projection_dim: int=128) -> None:
        super().__init__()
        hidden_dim = max(projection_dim * 2, input_dim // 2)
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError('Retrieval features must have shape [batch, frames, dim].')
        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=-1)
