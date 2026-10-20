"""Evaluation metrics implemented in NumPy."""
from __future__ import annotations

import numpy as np


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm  # rows = true class, columns = predicted class


def per_class_report(y_true, y_pred, class_names: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    report = {}
    for k, name in enumerate(class_names):
        tp = cm[k, k]
        precision = tp / max(1, cm[:, k].sum())
        recall = tp / max(1, cm[k, :].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        report[name] = {"precision": float(precision), "recall": float(recall),
                        "f1": float(f1), "support": int(cm[k, :].sum())}
    report["macro_f1"] = float(np.mean([report[n]["f1"] for n in class_names]))
    return report


def roc_auc(y_binary, scores) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic, with
    average ranks for ties: the probability a random positive outscores a
    random negative."""
    y = np.asarray(y_binary).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                       # assign average rank to ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def format_confusion(cm: np.ndarray, class_names: list[str]) -> str:
    w = max(len(n) for n in class_names) + 2
    head = "true \\ pred".ljust(w + 2) + "".join(n.rjust(w) for n in class_names)
    rows = [n.ljust(w + 2) + "".join(str(v).rjust(w) for v in row)
            for n, row in zip(class_names, cm)]
    return "\n".join([head] + rows)
