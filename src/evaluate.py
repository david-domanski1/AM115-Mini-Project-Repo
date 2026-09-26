"""Forecast scoring rules.  Outcomes are indexed 0 = home win, 1 = draw, 2 = away win."""
from __future__ import annotations

import numpy as np

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_to_index(outcomes) -> np.ndarray:
    return np.array([OUTCOME_INDEX[o] for o in outcomes], dtype=int)


def log_loss(probs, outcome_idx) -> float:
    """Mean negative log probability of the realised outcome, in nats per match."""
    probs = np.asarray(probs, float)
    p = probs[np.arange(len(outcome_idx)), outcome_idx]
    return float(-np.mean(np.log(np.clip(p, 1e-12, None))))


def brier(probs, outcome_idx) -> float:
    """Multi-class Brier score: mean over matches of sum_k (p_k - 1{k = outcome})^2 (range 0..2)."""
    probs = np.asarray(probs, float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcome_idx)), outcome_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def base_rate_probs(freq, n) -> np.ndarray:
    """Constant forecast repeated for n matches; ``freq`` = (pH, pD, pA)."""
    return np.tile(np.asarray(freq, float), (n, 1))
