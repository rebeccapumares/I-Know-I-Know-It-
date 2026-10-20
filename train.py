"""Train and evaluate the tip-of-the-tongue predictor.

Examples:
    python train.py                              # simulated multilingual dataset
    python train.py --data data/my_trials.csv    # your own data in the same format
"""
import argparse
import json
from pathlib import Path

import numpy as np

from totnet import (CLASS_NAMES, Adam, Preprocessor, SoftmaxCrossEntropy, accuracy, build_mlp,
                    confusion_matrix, encode_labels, gradient_check, load_csv, per_class_report,
                    roc_auc, save_csv, simulate_tot_dataset, split_by_participant)
from totnet.metrics import format_confusion

TOT = CLASS_NAMES.index("TOT")


def parse_args():
    p = argparse.ArgumentParser(description="Train the TOT predictor")
    p.add_argument("--data", default=None, help="CSV of trials; omit to simulate a dataset")
    p.add_argument("--participants", type=int, default=160)
    p.add_argument("--trials", type=int, default=90, help="trials per simulated participant")
    p.add_argument("--hidden", type=int, nargs="+", default=[32])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def evaluate(net, x, y) -> dict:
    proba = net.predict_proba(x)
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy(y, pred),
        "report": per_class_report(y, pred, CLASS_NAMES),
        "tot_auc": roc_auc(y == TOT, proba[:, TOT]),
        "confusion": confusion_matrix(y, pred, len(CLASS_NAMES)).tolist(),
    }


def permutation_importance(net, x, y, names, loss_fn, n_repeats: int = 5, seed: int = 0):
    """How much does test loss rise when one feature (or one-hot group) is
    shuffled? Bigger rise = the model relies on it more."""
    rng = np.random.default_rng(seed)
    base, _ = loss_fn(net.forward(x), y)
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(n.split("=")[0], []).append(i)
    scores = {}
    for g, cols in groups.items():
        rises = []
        for _ in range(n_repeats):
            xp = x.copy()
            xp[:, cols] = xp[rng.permutation(len(x))][:, cols]
            rises.append(loss_fn(net.forward(xp), y)[0] - base)
        scores[g] = float(np.mean(rises))
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def print_eval(title, res):
    r = res["report"]
    print(f"\n=== {title} ===")
    print(f"accuracy {res['accuracy']:.3f} | macro F1 {r['macro_f1']:.3f} | "
          f"TOT-vs-rest ROC AUC {res['tot_auc']:.3f}")
    for name in CLASS_NAMES:
        c = r[name]
        print(f"  {name:10s} precision {c['precision']:.3f}  recall {c['recall']:.3f}  "
              f"F1 {c['f1']:.3f}  (n={c['support']})")
    print(format_confusion(np.array(res["confusion"]), CLASS_NAMES))


def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. data
    if a.data:
        rows = load_csv(a.data)
        print(f"[data] loaded {len(rows):,} trials from {a.data}")
    else:
        rows = simulate_tot_dataset(a.participants, a.trials, seed=a.seed)
        Path("data").mkdir(exist_ok=True)
        save_csv(rows, "data/tot_trials_simulated.csv")
        print(f"[data] simulated {len(rows):,} trials -> data/tot_trials_simulated.csv")
    train_rows, val_rows, test_rows = split_by_participant(rows, seed=a.seed)
    prep = Preprocessor().fit(train_rows)
    x_tr, x_va, x_te = (prep.transform(r) for r in (train_rows, val_rows, test_rows))
    y_tr, y_va, y_te = (encode_labels(r) for r in (train_rows, val_rows, test_rows))
    counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
    print(f"[data] train/val/test trials: {len(y_tr)}/{len(y_va)}/{len(y_te)} "
          f"(split by participant) | {x_tr.shape[1]} features")
    print("[data] train class counts: " + ", ".join(f"{n}={c}" for n, c in zip(CLASS_NAMES, counts)))

    # 2. loss with inverse-frequency class weights so the minority TOT class counts
    weights = None if a.no_class_weights else counts.sum() / (len(counts) * counts)
    loss_fn = SoftmaxCrossEntropy(weights)

    # 3. verify backprop numerically before training
    probe = build_mlp(x_tr.shape[1], a.hidden, len(CLASS_NAMES), l2=a.l2, seed=a.seed)
    err = gradient_check(probe, loss_fn, x_tr[:32], y_tr[:32])
    print(f"[gradcheck] max relative error analytic vs numerical: {err:.2e} "
          f"({'PASS' if err < 1e-4 else 'FAIL'})")

    # 4. baseline: multinomial logistic regression = the same engine with no hidden layers
    print("\n[train] baseline: softmax regression (no hidden layers)")
    base = build_mlp(x_tr.shape[1], [], len(CLASS_NAMES), l2=a.l2, seed=a.seed)
    base.fit(x_tr, y_tr, loss_fn, Adam(lr=1e-2), epochs=a.epochs, batch_size=a.batch_size,
             x_val=x_va, y_val=y_va, seed=a.seed, verbose=False)

    # 5. the feedforward network
    print(f"[train] feedforward net {x_tr.shape[1]} -> {' -> '.join(map(str, a.hidden))} -> "
          f"{len(CLASS_NAMES)} (ReLU, dropout {a.dropout}, L2 {a.l2}, Adam lr {a.lr})")
    net = build_mlp(x_tr.shape[1], a.hidden, len(CLASS_NAMES), dropout=a.dropout, l2=a.l2,
                    seed=a.seed)
    history = net.fit(x_tr, y_tr, loss_fn, Adam(lr=a.lr), epochs=a.epochs,
                      batch_size=a.batch_size, x_val=x_va, y_val=y_va, seed=a.seed, log_every=10)

    # 6. evaluate on held-out participants
    majority = int(np.argmax(counts))
    res_major = {"accuracy": accuracy(y_te, np.full_like(y_te, majority))}
    res_base, res_net = evaluate(base, x_te, y_te), evaluate(net, x_te, y_te)
    print(f"\nmajority-class baseline accuracy: {res_major['accuracy']:.3f}")
    print_eval("softmax regression baseline (test)", res_base)
    print_eval("feedforward network (test)", res_net)

    importance = permutation_importance(net, x_te, y_te, prep.feature_names, loss_fn)
    print("\n=== permutation importance (rise in test loss) ===")
    for k, v in list(importance.items())[:8]:
        print(f"  {k:24s} {v:+.4f}")

    # 7. save
    net.save(out / "tot_model")
    (out / "preprocessor.json").write_text(json.dumps(prep.state()))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "gradcheck_max_rel_error": err, "majority_baseline": res_major,
        "softmax_regression": res_base, "feedforward_net": res_net,
        "permutation_importance": importance,
    }, indent=2))
    print(f"\n[save] model, preprocessor and metrics written to {out}/")


if __name__ == "__main__":
    main()
