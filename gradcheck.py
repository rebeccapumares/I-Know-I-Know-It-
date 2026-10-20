"""Numerical gradient checking: verifies the hand-written backprop against
centered finite differences, (L(w+eps) - L(w-eps)) / (2 eps)."""
from __future__ import annotations

import numpy as np


def gradient_check(net, loss_fn, x, y, eps: float = 1e-5, n_samples: int = 25,
                   seed: int = 0) -> float:
    """Return the worst relative error between analytic and numerical grads
    over a random sample of parameters in every trainable layer."""
    rng = np.random.default_rng(seed)
    net.train_mode(False)  # disable dropout so the loss is deterministic

    def loss_value():
        return loss_fn(net.forward(x), y)[0] + net.l2_penalty()

    _, grad = loss_fn(net.forward(x), y)
    net.backward(grad)
    analytic = [{k: g.copy() for k, g in l.grads.items()} for l in net.layers]

    worst = 0.0
    for li, layer in enumerate(net.layers):
        for name, p in layer.params.items():
            flat = p.reshape(-1)
            for j in rng.choice(flat.size, size=min(n_samples, flat.size), replace=False):
                old = flat[j]
                flat[j] = old + eps
                plus = loss_value()
                flat[j] = old - eps
                minus = loss_value()
                flat[j] = old
                numeric = (plus - minus) / (2 * eps)
                a = analytic[li][name].reshape(-1)[j]
                rel = abs(a - numeric) / max(1e-8, abs(a) + abs(numeric))
                worst = max(worst, rel)
    return worst
