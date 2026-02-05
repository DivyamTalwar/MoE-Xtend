from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class SamplingConfig:
    temperature: float = 0.1
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalize_prompt: bool = True
    logit_bias: Dict[int, float] = field(default_factory=dict)


def apply_logit_bias(logits: torch.Tensor, logit_bias: Dict[int, float]) -> None:
    if not logit_bias:
        return
    ids = torch.tensor(list(logit_bias.keys()), device=logits.device, dtype=torch.long)
    bias = torch.tensor(list(logit_bias.values()), device=logits.device, dtype=logits.dtype)
    logits.index_add_(0, ids, bias)


def apply_penalties(
    logits: torch.Tensor,
    token_counts: Dict[int, int],
    *,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
) -> None:
    if not token_counts:
        return

    ids = torch.tensor(list(token_counts.keys()), device=logits.device, dtype=torch.long)
    counts = torch.tensor(list(token_counts.values()), device=logits.device, dtype=logits.dtype)
    selected = logits.index_select(0, ids)

    if repetition_penalty != 1.0:
        selected = torch.where(selected < 0, selected * repetition_penalty, selected / repetition_penalty)

    if frequency_penalty != 0.0:
        selected = selected - frequency_penalty * counts

    if presence_penalty != 0.0:
        selected = selected - presence_penalty * (counts > 0).to(selected.dtype)

    logits.index_copy_(0, ids, selected)


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, top_k)
    cutoff = values[-1]
    return torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)

    cutoff = cumprobs > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))
    return logits.scatter(0, sorted_indices, sorted_logits)


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max()
    return torch.where(probs < min_p * max_prob, torch.full_like(logits, -float("inf")), logits)


def sample_token(logits: torch.Tensor, cfg: SamplingConfig) -> int:
    if cfg.temperature == 0.0:
        return torch.argmax(logits, dim=-1).item()

    logits = logits / max(cfg.temperature, 1e-6)
    logits = top_k_filter(logits, cfg.top_k)
    logits = top_p_filter(logits, cfg.top_p)
    logits = min_p_filter(logits, cfg.min_p)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
