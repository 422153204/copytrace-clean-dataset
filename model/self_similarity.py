from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSimilarity(nn.Module):

    def __init__(self, temperature: float=0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError('Feature inputs must have shape [batch, frames, dim].')
        normalized = F.normalize(features, p=2, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(1, 2))
        return similarity / self.temperature

class SimilarityRidgeEncoder(nn.Module):

    def __init__(self, hidden_channels: int=32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        if similarity.ndim != 3:
            raise ValueError('Similarity matrices must have shape [batch, frames, frames].')
        return self.net(similarity.unsqueeze(1))
