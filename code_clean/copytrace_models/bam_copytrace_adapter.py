from __future__ import annotations
import torch
import torch.nn as nn
from .copytrace_ssl import CopyTraceSSL

class BamCopyTraceAdapter(nn.Module):

    def __init__(self, bam_backbone: nn.Module, frame_dim: int, ridge_channels: int=32) -> None:
        super().__init__()
        self.bam_backbone = bam_backbone
        self.copytrace = CopyTraceSSL(frame_dim=frame_dim, ridge_channels=ridge_channels)

    def forward(self, waveform: torch.Tensor) -> dict:
        frame_embeddings = self.bam_backbone(waveform, ret_emb=True)
        outputs = self.copytrace(frame_embeddings)
        outputs['frame_embeddings'] = frame_embeddings
        return outputs
