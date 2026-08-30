from __future__ import annotations
import torch
import torch.nn.functional as F

def mask_interval(mask: torch.Tensor, threshold: float=0.5) -> tuple[int, int] | None:
    indices = torch.nonzero(mask > threshold, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    return (int(indices[0].item()), int(indices[-1].item()) + 1)

def ordered_candidate_scores(target_sequence: torch.Tensor, source_embeddings: torch.Tensor, alignment_radius: int=1) -> torch.Tensor:
    if target_sequence.ndim != 2 or source_embeddings.ndim != 2:
        raise ValueError('目标序列和源嵌入必须使用 [帧数, 维度] 形状。')
    if target_sequence.shape[1] != source_embeddings.shape[1]:
        raise ValueError('目标序列与源嵌入的特征维度必须一致。')
    target_length = target_sequence.shape[0]
    source_length = source_embeddings.shape[0]
    if target_length <= 0 or target_length > source_length:
        return source_embeddings.new_empty((0,))
    candidate_count = source_length - target_length + 1
    radius = max(0, int(alignment_radius))
    similarity = torch.matmul(target_sequence, source_embeddings.transpose(0, 1))
    candidate_starts = torch.arange(candidate_count, device=source_embeddings.device).reshape(-1, 1, 1)
    target_offsets = torch.arange(target_length, device=source_embeddings.device).reshape(1, -1, 1)
    local_offsets = torch.arange(-radius, radius + 1, device=source_embeddings.device).reshape(1, 1, -1)
    source_indices = candidate_starts + target_offsets + local_offsets
    valid = (source_indices >= 0) & (source_indices < source_length)
    source_indices = source_indices.clamp(0, source_length - 1)
    target_indices = torch.arange(target_length, device=source_embeddings.device).reshape(1, -1, 1)
    target_indices = target_indices.expand(candidate_count, -1, 2 * radius + 1)
    local_similarity = similarity[target_indices, source_indices]
    local_similarity = local_similarity.masked_fill(~valid, float('-inf'))
    strict_similarity = local_similarity[:, :, radius]
    if radius > 0:
        tolerant_similarity = local_similarity.max(dim=-1).values
        aligned_similarity = 0.75 * strict_similarity + 0.25 * tolerant_similarity
    else:
        aligned_similarity = strict_similarity
    return aligned_similarity.mean(dim=-1)

def pooled_candidate_scores(target_sequence: torch.Tensor, source_embeddings: torch.Tensor) -> torch.Tensor:
    target_length = target_sequence.shape[0]
    source_length = source_embeddings.shape[0]
    if target_length <= 0 or target_length > source_length:
        return source_embeddings.new_empty((0,))
    query = F.normalize(target_sequence.mean(dim=0), p=2, dim=0)
    frame_scores = torch.matmul(source_embeddings, query)
    return F.avg_pool1d(frame_scores.reshape(1, 1, -1), kernel_size=target_length, stride=1).reshape(-1)

def mask_overlapping_candidates(scores: torch.Tensor, frame_exclusion: torch.Tensor | None, segment_length: int, padding: int=0) -> torch.Tensor:
    if frame_exclusion is None or scores.numel() == 0:
        return scores
    exclusion = frame_exclusion.to(device=scores.device, dtype=torch.float32)
    padding = max(0, int(padding))
    if padding > 0:
        exclusion = F.max_pool1d(exclusion.reshape(1, 1, -1), kernel_size=2 * padding + 1, stride=1, padding=padding).reshape(-1)
    segment_length = max(1, min(int(segment_length), exclusion.numel()))
    candidate_exclusion = F.max_pool1d(exclusion.reshape(1, 1, -1), kernel_size=segment_length, stride=1).reshape(-1) > 0
    candidate_exclusion = candidate_exclusion[:scores.numel()]
    return scores.masked_fill(candidate_exclusion, float('-inf'))

def restrict_to_coarse_top_k(ordered_scores: torch.Tensor, coarse_scores: torch.Tensor, top_k: int, required_start: int | None=None, required_radius: int=0) -> torch.Tensor:
    if top_k <= 0 or top_k >= ordered_scores.numel():
        return ordered_scores
    if ordered_scores.numel() != coarse_scores.numel():
        raise ValueError('有序候选分数和粗筛分数数量必须一致。')
    finite = torch.isfinite(coarse_scores)
    finite_count = int(finite.sum().item())
    if finite_count <= top_k:
        return ordered_scores
    safe_coarse = coarse_scores.masked_fill(~finite, float('-inf'))
    keep = torch.zeros_like(finite)
    selected = torch.topk(safe_coarse, k=min(top_k, finite_count)).indices
    keep[selected] = True
    if required_start is not None:
        radius = max(0, int(required_radius))
        start = max(0, int(required_start) - radius)
        end = min(keep.numel(), int(required_start) + radius + 1)
        keep[start:end] = True
    return ordered_scores.masked_fill(~keep, float('-inf'))

def soft_start_targets(candidate_count: int, true_start: int, tolerance: int, reference: torch.Tensor) -> torch.Tensor:
    if candidate_count <= 0:
        return reference.new_empty((0,))
    starts = torch.arange(candidate_count, device=reference.device)
    distance = (starts - int(true_start)).abs().to(reference.dtype)
    tolerance = max(0, int(tolerance))
    weights = (tolerance + 1.0 - distance).clamp_min(0.0)
    if float(weights.sum().item()) <= 0.0:
        nearest = min(max(0, int(true_start)), candidate_count - 1)
        weights[nearest] = 1.0
    return weights / weights.sum().clamp_min(1e-08)

def best_candidate_mask(scores: torch.Tensor, total_frames: int, segment_length: int) -> tuple[torch.Tensor, int | None]:
    predicted = torch.zeros(total_frames, dtype=torch.bool, device=scores.device)
    finite = torch.isfinite(scores)
    if scores.numel() == 0 or not bool(finite.any()):
        return (predicted, None)
    best_start = int(torch.argmax(scores).item())
    best_end = min(total_frames, best_start + max(1, int(segment_length)))
    predicted[best_start:best_end] = True
    return (predicted, best_start)
