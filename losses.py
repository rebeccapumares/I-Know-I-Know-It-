"""Loss functions. Each returns (loss, dL/dlogits) so backprop can start."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)          # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax + (optionally class-weighted) cross-entropy, fused.

    Fusing the two gives the clean gradient dL/dz = (softmax(z) - onehot(y)) / N,
    which avoids computing the full softmax Jacobian. With class weights w,
    each row's gradient is scaled by w[y_i] and normalized by sum(w[y]).
    """

    def __init__(self, class_weights: np.ndarray | None = None):
        self.class_weights = None if class_weights is None else np.asarray(class_weights, float)

    def __call__(self, logits: np.ndarray, y: np.ndarray):
        n, k = logits.shape
        probs = softmax(logits)
        w = np.ones(n) if self.class_weights is None else self.class_weights[y]
        norm = w.sum()
        loss = -(w * np.log(probs[np.arange(n), y] + 1e-12)).sum() / norm
        grad = probs.copy()
        grad[np.arange(n), y] -= 1.0
        grad *= (w / norm)[:, None]
        return loss, grad


class BinaryCrossEntropy:
    """Sigmoid + binary cross-entropy on a single logit column, fused."""

    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = pos_weight

    def __call__(self, logits: np.ndarray, y: np.ndarray):
        z = logits.reshape(-1)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        w = np.where(y == 1, self.pos_weight, 1.0)
        norm = w.sum()
        loss = -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / norm
        grad = (w * (p - y) / norm).reshape(logits.shape)
        return loss, grad
