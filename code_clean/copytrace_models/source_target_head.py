from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn

class TemporalResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        group_count = 8 if hidden_dim % 8 == 0 else 1
        self.block = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden_dim), nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1), nn.GroupNorm(group_count, hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)

class TemporalTargetRefinementHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=128, dropout: float=0.1) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.temporal_blocks = nn.ModuleList([TemporalResidualBlock(hidden_dim, dilation=1, dropout=dropout), TemporalResidualBlock(hidden_dim, dilation=2, dropout=dropout), TemporalResidualBlock(hidden_dim, dilation=4, dropout=dropout)])
        self.output_projection = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features).transpose(1, 2)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        return self.output_projection(hidden).squeeze(1)

class TargetPresenceHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=64) -> None:
        super().__init__()
        self.feature_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.output_projection = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_projection(features)
        attention = self.attention_score(hidden).squeeze(-1).softmax(dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        return self.output_projection(pooled).squeeze(-1)

@dataclass(frozen=True)
class SourceTargetPrediction:
    target_logits: torch.Tensor
    source_logits: torch.Tensor
    boundary_logits: torch.Tensor
    pair_logits: torch.Tensor
    direction_logits: torch.Tensor
    target_presence_logits: Optional[torch.Tensor] = None

class SourceTargetHead(nn.Module):

    def __init__(self, frame_dim: int, ridge_channels: int=32, use_boundary: bool=True, use_direction: bool=True, use_pair: bool=True, use_temporal_target_head: bool=False, temporal_target_hidden_dim: int=128, use_target_presence_head: bool=False, target_presence_hidden_dim: int=64) -> None:
        super().__init__()
        self.target_head = nn.Linear(frame_dim, 1)
        self.temporal_target_head = TemporalTargetRefinementHead(input_dim=frame_dim + 3, hidden_dim=temporal_target_hidden_dim) if use_temporal_target_head else None
        self.target_presence_head = TargetPresenceHead(input_dim=frame_dim + 3, hidden_dim=target_presence_hidden_dim) if use_target_presence_head else None
        self.source_head = nn.Sequential(nn.Linear(frame_dim + 3, frame_dim // 2), nn.ReLU(inplace=True), nn.Linear(frame_dim // 2, 1))
        if use_boundary:
            self.boundary_head = nn.Linear(frame_dim, 1)
        else:
            self.boundary_head = None
        if use_pair and ridge_channels > 0:
            self.pair_head = nn.Conv2d(ridge_channels, 1, kernel_size=1)
        else:
            self.pair_head = None
        if use_direction:
            self.direction_head = nn.Sequential(nn.Linear(frame_dim * 4, frame_dim), nn.ReLU(inplace=True), nn.Linear(frame_dim, 1))
        else:
            self.direction_head = None

    def forward(self, frame_features: torch.Tensor, ridge_features: Optional[torch.Tensor], left_context: torch.Tensor, right_context: torch.Tensor, src_sim_score: Optional[torch.Tensor]=None) -> SourceTargetPrediction:
        B, T, D = frame_features.shape
        if self.boundary_head is not None:
            boundary_logits = self.boundary_head(frame_features).squeeze(-1)
        else:
            boundary_logits = torch.zeros(B, T, device=frame_features.device)
        if self.direction_head is not None:
            direction_features = torch.cat([frame_features, left_context, right_context, (left_context - right_context).abs()], dim=-1)
            direction_logits = self.direction_head(direction_features).squeeze(-1)
        else:
            direction_logits = torch.zeros(B, T, device=frame_features.device)
        target_logits = self.target_head(frame_features).squeeze(-1)
        target_features = None
        if self.temporal_target_head is not None or self.target_presence_head is not None:
            similarity_score = src_sim_score if src_sim_score is not None else torch.zeros(B, T, device=frame_features.device)
            target_features = torch.cat([frame_features, similarity_score.unsqueeze(-1), boundary_logits.sigmoid().unsqueeze(-1), direction_logits.sigmoid().unsqueeze(-1)], dim=-1)
        if self.temporal_target_head is not None and target_features is not None:
            target_logits = target_logits + self.temporal_target_head(target_features)
        target_presence_logits = self.target_presence_head(target_features) if self.target_presence_head is not None and target_features is not None else None
        if src_sim_score is not None:
            boundary_score = boundary_logits.sigmoid()
            direction_score = direction_logits.sigmoid()
            src_input = torch.cat([frame_features, src_sim_score.unsqueeze(-1), boundary_score.unsqueeze(-1), direction_score.unsqueeze(-1)], dim=-1)
        else:
            src_input = torch.cat([frame_features, torch.zeros(B, T, 3, device=frame_features.device)], dim=-1)
        source_logits = self.source_head(src_input).squeeze(-1)
        if self.pair_head is not None and ridge_features is not None:
            pair_logits = self.pair_head(ridge_features).squeeze(1)
        else:
            pair_logits = torch.zeros(B, T, T, device=frame_features.device)
        return SourceTargetPrediction(target_logits=target_logits, source_logits=source_logits, boundary_logits=boundary_logits, pair_logits=pair_logits, direction_logits=direction_logits, target_presence_logits=target_presence_logits)

def prediction_to_dict(prediction: SourceTargetPrediction) -> Dict[str, torch.Tensor]:
    outputs = {'target_logits': prediction.target_logits, 'source_logits': prediction.source_logits, 'boundary_logits': prediction.boundary_logits, 'pair_logits': prediction.pair_logits, 'direction_logits': prediction.direction_logits}
    if prediction.target_presence_logits is not None:
        outputs['target_presence_logits'] = prediction.target_presence_logits
    return outputs
