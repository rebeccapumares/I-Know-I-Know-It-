"""Feedforward network: chains layers, runs backpropagation, trains with
mini-batches, early stopping and learning-rate decay, and saves/loads weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .layers import LAYER_TYPES, Dense, Dropout, Layer
from .losses import softmax


class FeedforwardNet:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    # ------------------------------------------------------------ passes
    def train_mode(self, on: bool = True):
        for layer in self.layers:
            layer.training = on

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagation: push dL/d(output) back through every layer in
        reverse order (chain rule), filling each layer's parameter grads."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def l2_penalty(self) -> float:
        return sum(0.5 * l.l2 * (l.params["W"] ** 2).sum()
                   for l in self.layers if isinstance(l, Dense) and l.l2 > 0)

    # --------------------------------------------------------- inference
    def predict_proba(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.train_mode(False)
        out = [softmax(self.forward(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]
        return np.vstack(out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    # ---------------------------------------------------------- training
    def fit(self, x, y, loss_fn, optimizer, epochs: int = 100, batch_size: int = 64,
            x_val=None, y_val=None, patience: int = 15, lr_decay: float = 0.5,
            decay_patience: int = 6, min_lr: float = 1e-5, seed: int = 0,
            verbose: bool = True, log_every: int = 5) -> list[dict]:
        rng = np.random.default_rng(seed)
        history, best_val, best_weights = [], np.inf, None
        stale, stale_decay = 0, 0

        for epoch in range(1, epochs + 1):
            self.train_mode(True)
            order = rng.permutation(len(x))
            total, seen = 0.0, 0
            for start in range(0, len(x), batch_size):
                idx = order[start:start + batch_size]
                logits = self.forward(x[idx])
                loss, grad = loss_fn(logits, y[idx])
                self.backward(grad)
                optimizer.step(self.layers)
                total += loss * len(idx)
                seen += len(idx)

            record = {"epoch": epoch, "train_loss": total / seen + self.l2_penalty(),
                      "lr": optimizer.lr}
            if x_val is not None:
                self.train_mode(False)
                val_loss, _ = loss_fn(self.forward(x_val), y_val)
                record["val_loss"] = val_loss + self.l2_penalty()
                record["val_acc"] = float((self.predict(x_val) == y_val).mean())
                # early stopping + reduce-on-plateau
                if record["val_loss"] < best_val - 1e-4:
                    best_val, best_weights = record["val_loss"], self.get_weights()
                    stale = stale_decay = 0
                else:
                    stale += 1
                    stale_decay += 1
                    if stale_decay >= decay_patience and optimizer.lr > min_lr:
                        optimizer.lr = max(min_lr, optimizer.lr * lr_decay)
                        stale_decay = 0
            history.append(record)

            if verbose and (epoch == 1 or epoch % log_every == 0):
                msg = f"epoch {epoch:3d} | train {record['train_loss']:.4f}"
                if "val_loss" in record:
                    msg += f" | val {record['val_loss']:.4f} | val acc {record['val_acc']:.3f}"
                print(msg + f" | lr {optimizer.lr:.1e}")
            if x_val is not None and stale >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                break

        if best_weights is not None:
            self.set_weights(best_weights)
        self.train_mode(False)
        return history

    # ------------------------------------------------------- persistence
    def get_weights(self):
        return [{k: v.copy() for k, v in l.params.items()} for l in self.layers]

    def set_weights(self, weights):
        for layer, w in zip(self.layers, weights):
            for k, v in w.items():
                layer.params[k][...] = v

    def save(self, path: str | Path):
        path = Path(path)
        arrays = {f"{i}_{k}": v for i, l in enumerate(self.layers) for k, v in l.params.items()}
        np.savez(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps([l.config() for l in self.layers], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeedforwardNet":
        path = Path(path)
        configs = json.loads(path.with_suffix(".json").read_text())
        arrays = np.load(path.with_suffix(".npz"))
        layers = []
        for i, cfg in enumerate(configs):
            kind = cfg.pop("type")
            layer = LAYER_TYPES[kind](**cfg)
            for k in layer.params:
                layer.params[k] = arrays[f"{i}_{k}"]
            layers.append(layer)
        net = cls(layers)
        net.train_mode(False)
        return net


def build_mlp(n_in: int, hidden: list[int], n_out: int, dropout: float = 0.0,
              l2: float = 0.0, seed: int = 0) -> FeedforwardNet:
    """Dense -> ReLU (-> Dropout) for each hidden width, then a Dense output."""
    from .layers import ReLU
    rng = np.random.default_rng(seed)
    layers: list[Layer] = []
    prev = n_in
    for width in hidden:
        layers += [Dense(prev, width, init="he", l2=l2, rng=rng), ReLU()]
        if dropout > 0:
            layers.append(Dropout(dropout, rng=rng))
        prev = width
    layers.append(Dense(prev, n_out, init="xavier", l2=l2, rng=rng))
    return FeedforwardNet(layers)
