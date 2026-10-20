"""Optimizers that update layer parameters in place from their gradients."""
from __future__ import annotations

import numpy as np


class SGD:
    """Stochastic gradient descent with optional (Nesterov-free) momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr, self.momentum = lr, momentum
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            for name, p in layer.params.items():
                key = (i, name)
                v = self.velocity.get(key, np.zeros_like(p))
                v = self.momentum * v - self.lr * layer.grads[name]
                self.velocity[key] = v
                p += v


class Adam:
    """Adam: per-parameter adaptive step sizes from running moments of the
    gradient (m) and squared gradient (v), with bias correction."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m: dict[tuple[int, str], np.ndarray] = {}
        self.v: dict[tuple[int, str], np.ndarray] = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for i, layer in enumerate(layers):
            for name, p in layer.params.items():
                key, g = (i, name), layer.grads[name]
                m = b1 * self.m.get(key, np.zeros_like(p)) + (1 - b1) * g
                v = b2 * self.v.get(key, np.zeros_like(p)) + (1 - b2) * g * g
                self.m[key], self.v[key] = m, v
                m_hat = m / (1 - b1 ** self.t)
                v_hat = v / (1 - b2 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
