"""Neural network layers with hand-written forward and backward passes.

Every layer caches what it needs during `forward` and returns the gradient
with respect to its input from `backward(grad_out)`. Trainable layers store
parameter gradients in `self.grads`, keyed like `self.params`.
"""
from __future__ import annotations

import numpy as np


class Layer:
    trainable = False

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        return {"type": type(self).__name__}


class Dense(Layer):
    """Fully connected layer: y = x W + b.

    Backward pass (for a batch of N rows):
        dL/dW = x^T  . dL/dy
        dL/db = sum over batch of dL/dy
        dL/dx = dL/dy . W^T
    """
    trainable = True

    def __init__(self, n_in: int, n_out: int, init: str = "he", l2: float = 0.0,
                 rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng()
        if init == "he":          # suited to ReLU
            scale = np.sqrt(2.0 / n_in)
        elif init == "xavier":    # suited to tanh / sigmoid
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f"unknown init: {init}")
        self.n_in, self.n_out, self.init, self.l2 = n_in, n_out, init, l2
        self.params = {"W": rng.normal(0.0, scale, (n_in, n_out)), "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        self.grads["W"] = self.x.T @ grad_out + self.l2 * self.params["W"]
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T

    def config(self):
        return {"type": "Dense", "n_in": self.n_in, "n_out": self.n_out,
                "init": self.init, "l2": self.l2}


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_out):
        return grad_out * np.where(self.x > 0, 1.0, self.alpha)

    def config(self):
        return {"type": "LeakyReLU", "alpha": self.alpha}


class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1.0 - self.out ** 2)


class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1.0 - self.out)


class Dropout(Layer):
    """Inverted dropout: active only in training mode, identity at inference."""

    def __init__(self, rate: float = 0.2, rng: np.random.Generator | None = None):
        super().__init__()
        self.rate = rate
        self.rng = rng or np.random.default_rng()

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.rate) / (1.0 - self.rate)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out if self.mask is None else grad_out * self.mask

    def config(self):
        return {"type": "Dropout", "rate": self.rate}


LAYER_TYPES = {cls.__name__: cls for cls in (Dense, ReLU, LeakyReLU, Tanh, Sigmoid, Dropout)}
