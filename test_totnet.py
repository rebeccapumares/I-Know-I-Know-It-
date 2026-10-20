"""Tests for the from-scratch network engine. Run with: python -m pytest tests -q"""
import numpy as np

from totnet import (CLASS_NAMES, SGD, Adam, BinaryCrossEntropy, Dense, Dropout, FeedforwardNet,
                    LeakyReLU, Preprocessor, Sigmoid, SoftmaxCrossEntropy, Tanh, build_mlp,
                    encode_labels, gradient_check, roc_auc, simulate_tot_dataset,
                    split_by_participant)

rng = np.random.default_rng(0)


def test_gradcheck_relu_mlp_with_l2_and_class_weights():
    net = build_mlp(6, [8, 5], 3, l2=1e-2, seed=1)
    x, y = rng.normal(size=(16, 6)), rng.integers(0, 3, 16)
    assert gradient_check(net, SoftmaxCrossEntropy(np.array([1.0, 2.5, 0.7])), x, y) < 1e-5


def test_gradcheck_other_activations_and_bce():
    r = np.random.default_rng(2)
    net = FeedforwardNet([Dense(4, 6, init="xavier", rng=r), Tanh(),
                          Dense(6, 5, rng=r), LeakyReLU(0.1),
                          Dense(5, 4, rng=r), Sigmoid(), Dense(4, 1, rng=r)])
    x, y = r.normal(size=(12, 4)), r.integers(0, 2, 12)
    assert gradient_check(net, BinaryCrossEntropy(pos_weight=3.0), x, y) < 1e-5


def test_softmax_ce_matches_definition():
    logits, y = rng.normal(size=(5, 3)), np.array([0, 2, 1, 1, 0])
    loss, _ = SoftmaxCrossEntropy()(logits, y)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    assert np.isclose(loss, -np.log(p[np.arange(5), y]).mean())


def test_dropout_is_identity_at_inference_and_unbiased_in_training():
    d = Dropout(0.5, rng=np.random.default_rng(0))
    x = np.ones((2000, 50))
    d.training = False
    assert np.array_equal(d.forward(x), x)
    d.training = True
    assert abs(d.forward(x).mean() - 1.0) < 0.02


def test_learns_xor_nonlinear():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    net = build_mlp(2, [8], 2, seed=3)
    net.fit(np.repeat(x, 25, 0), np.repeat(y, 25), SoftmaxCrossEntropy(), Adam(lr=0.05),
            epochs=200, batch_size=20, verbose=False)
    assert (net.predict(x) == y).all()


def test_sgd_reduces_loss():
    x = rng.normal(size=(300, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    net = build_mlp(4, [8], 2, seed=0)
    loss_fn = SoftmaxCrossEntropy()
    before = loss_fn(net.forward(x), y)[0]
    net.fit(x, y, loss_fn, SGD(lr=0.05), epochs=30, verbose=False)
    assert loss_fn(net.forward(x), y)[0] < before * 0.5


def test_roc_auc():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_save_load_roundtrip(tmp_path):
    net = build_mlp(5, [7], 3, dropout=0.2, seed=4)
    x = rng.normal(size=(10, 5))
    net.save(tmp_path / "m")
    loaded = FeedforwardNet.load(tmp_path / "m")
    assert np.allclose(net.predict_proba(x), loaded.predict_proba(x))


def test_data_pipeline_no_participant_leakage():
    rows = simulate_tot_dataset(n_participants=30, trials_per_participant=20, seed=1)
    tr, va, te = split_by_participant(rows)
    ids = [{r["participant_id"] for r in s} for s in (tr, va, te)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    prep = Preprocessor().fit(tr)
    x = prep.transform(te)
    assert x.shape == (len(te), len(prep.feature_names)) and np.isfinite(x).all()
    assert set(encode_labels(rows)) == set(range(len(CLASS_NAMES)))


def test_simulated_bilingual_tot_effect():
    rows = simulate_tot_dataset(n_participants=80, trials_per_participant=60, seed=3)
    rate = lambda role: np.mean([r["outcome"] == "TOT" for r in rows
                                 if r["test_language_role"] == role])
    assert rate("nondominant") > rate("dominant")
