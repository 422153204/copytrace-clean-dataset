from __future__ import annotations
import torch
import torch.nn as nn
from .self_similarity import SelfSimilarity, SimilarityRidgeEncoder
from .source_target_head import SourceTargetHead, prediction_to_dict
from .global_retrieval import GlobalSourceRetrievalHead

class CopyTraceSSL(nn.Module):

    def __init__(self, frame_dim: int, similarity_temperature: float=0.07, ridge_channels: int=32, use_selfsim: bool=True, use_boundary: bool=True, use_direction: bool=True, use_pair: bool=True, retrieval_dim: int=128, use_temporal_target_head: bool=False, temporal_target_hidden_dim: int=128, use_target_presence_head: bool=False, target_presence_hidden_dim: int=64) -> None:
        super().__init__()
        self.use_selfsim = use_selfsim
        self.use_boundary = use_boundary
        self.use_direction = use_direction
        self.use_pair = use_pair
        self.frame_dim = frame_dim
        self.global_retrieval_head = GlobalSourceRetrievalHead(input_dim=frame_dim, projection_dim=retrieval_dim)
        if use_selfsim:
            self.self_similarity = SelfSimilarity(temperature=similarity_temperature)
            if use_pair:
                self.ridge_encoder = SimilarityRidgeEncoder(hidden_channels=ridge_channels)
            else:
                self.ridge_encoder = None
        else:
            self.self_similarity = None
            self.ridge_encoder = None
        self.head = SourceTargetHead(frame_dim=frame_dim, ridge_channels=ridge_channels, use_boundary=use_boundary, use_direction=use_direction, use_pair=use_pair, use_temporal_target_head=use_temporal_target_head, temporal_target_hidden_dim=temporal_target_hidden_dim, use_target_presence_head=use_target_presence_head, target_presence_hidden_dim=target_presence_hidden_dim)

    def forward(self, frame_features: torch.Tensor, retrieval_features: torch.Tensor | None=None) -> dict:
        outputs: dict = {}
        if self.use_selfsim and self.self_similarity is not None:
            similarity = self.self_similarity(frame_features)
            outputs['similarity'] = similarity
            T = similarity.shape[-1]
            mask = 1.0 - torch.eye(T, device=similarity.device).unsqueeze(0)
            sim_no_diag = similarity * mask
            src_sim_score = sim_no_diag.max(dim=-1).values
            if self.use_pair and self.ridge_encoder is not None:
                ridge_features = self.ridge_encoder(similarity)
            else:
                ridge_features = None
        else:
            similarity = None
            ridge_features = None
            src_sim_score = None
        left_context, right_context = self._neighbor_context(frame_features)
        prediction = self.head(frame_features, ridge_features, left_context, right_context, src_sim_score=src_sim_score)
        outputs.update(prediction_to_dict(prediction))
        outputs['source_pair_logits'] = self._source_from_pair_logits(outputs['pair_logits'], outputs['target_logits'])
        retrieval_input = frame_features if retrieval_features is None else retrieval_features
        outputs['retrieval_embeddings'] = self.global_retrieval_head(retrieval_input)
        if not self.use_boundary:
            outputs['boundary_logits'] = torch.zeros_like(outputs['target_logits'])
        if not self.use_direction:
            outputs['direction_logits'] = torch.zeros_like(outputs['target_logits'])
        return outputs

    def encode_retrieval(self, retrieval_features: torch.Tensor) -> torch.Tensor:
        return self.global_retrieval_head(retrieval_features)

    @staticmethod
    def _neighbor_context(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.roll(features, shifts=1, dims=1)
        right = torch.roll(features, shifts=-1, dims=1)
        left[:, 0, :] = features[:, 0, :]
        right[:, -1, :] = features[:, -1, :]
        return (left, right)

    @staticmethod
    def _source_from_pair_logits(pair_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        if pair_logits.ndim != 3 or target_logits.ndim != 2:
            return torch.zeros_like(target_logits)
        T = min(pair_logits.shape[1], pair_logits.shape[2], target_logits.shape[1])
        pair_logits = pair_logits[:, :T, :T]
        target_logits = target_logits[:, :T]
        target_prob = target_logits.sigmoid().detach()
        target_peak = target_prob.max(dim=-1, keepdim=True).values
        adaptive_threshold = torch.maximum(torch.full_like(target_peak, 0.1), target_peak * 0.6)
        target_mask = target_prob >= adaptive_threshold
        masked_pair = pair_logits.masked_fill(~target_mask.unsqueeze(1), -20.0)
        source_logits = masked_pair.max(dim=-1).values
        no_target = target_peak.squeeze(-1) < 0.1
        if no_target.any():
            source_logits = source_logits.clone()
            source_logits[no_target] = -20.0
        return source_logits

class MFCCFeatureExtractor(nn.Module):

    def __init__(self, n_mfcc: int=40, n_fft: int=400, hop_length: int=320):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        import torchaudio
        self.transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc, melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': max(80, n_mfcc)})

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        mfcc = self.transform(waveforms)
        return mfcc.transpose(1, 2)
