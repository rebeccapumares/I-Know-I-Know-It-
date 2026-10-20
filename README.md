# I Know I Know It — Tip-of-the-Tongue Predictor

A feedforward neural network and backpropagation engine built **from scratch in NumPy**,
with no PyTorch, TensorFlow or scikit-learn. It predicts **tip-of-the-tongue (TOT)
states**: the "I know it, I know it…" feeling of being sure you know a word while being
unable to retrieve it. Predictions are made from multilingual recall, response-time and
recognition data.

![Interactive demo](web/screenshot.png)

**Interactive demo:** open `web/index.html` in a browser. It runs the trained network with its
exact weights in JavaScript (verified to match the NumPy predictions), so you can adjust a recall
trial and watch the prediction, hidden-layer activations and per-input explanations update live.

Each trial is classified as one of three retrieval states:

| Class | Meaning |
|---|---|
| `KNOW` | the word was retrieved |
| `TOT` | the word feels known but won't come out |
| `DONT_KNOW` | the word isn't known |

## What's implemented from scratch

**Neural network engine (`totnet/`)**
- `Dense` layers with He / Xavier initialization and L2 regularization
- Activations: `ReLU`, `LeakyReLU`, `Tanh`, `Sigmoid`, plus inverted `Dropout`
- **Backpropagation**: every layer has a hand-derived backward pass, and the network chains
  them in reverse using the chain rule
- Losses: fused softmax + cross-entropy (gradient `p − y`) with class weighting for the
  minority TOT class, and binary cross-entropy
- Optimizers: SGD with momentum and Adam with bias correction
- Training loop: mini-batches, early stopping, reduce-on-plateau learning-rate decay, and
  best-weights restore
- **Numerical gradient checking**: analytic gradients are verified against finite
  differences before every training run, with a relative error around 1e-7
- Metrics: confusion matrix, per-class precision / recall / F1, and ROC AUC (Mann–Whitney)
- Permutation feature importance, plus model save / load

**Data pipeline (`totnet/data.py`)**
- Log-transformed and z-scored response times, binary indicators, and one-hot language
  features
- **Participant-level train / val / test split**, so the model is tested on people it has
  never seen and can't memorize individual habits

## Features used (22 after encoding)

| Group | Features |
|---|---|
| Multilingual | test language, language role (dominant / non-dominant / third), proficiency, cognate status, translation recalled in another language |
| Response time | recall RT, recognition RT |
| Recognition | later recognition test correct |
| Partial recall | first letter reported, syllable count guessed, feeling-of-knowing rating (1–6) |
| Item / person | word frequency, age of acquisition, syllables, proper name, participant age |

Whether the recall attempt itself succeeded is deliberately **not** a feature, since that
would make the task trivial.

## Data

> **Note:** the bundled dataset (`data/tot_trials_simulated.csv`, 14,400 trials from 160
> simulated participants) is **synthetic**. No public multilingual TOT dataset with this
> combination of measures is readily available, so `simulate_tot_dataset` generates one
> whose structure follows well-replicated TOT findings.

Effects built into the simulator:

- Rarer words and later-acquired words produce more TOTs
- **Bilingual TOT effect**: more TOTs in a non-dominant or third language
  (17% → 26% → 32% in the generated data)
- Cognates reduce TOTs; proper names and older age increase them
- TOTs come with long searches, a high feeling of knowing, partial access to the sound of the
  word (first letter, syllables), and good later recognition
- Large individual differences: people vary in response speed, rating style and how long
  they persist, so the same state looks different across participants

To use real data, pass a CSV with the same columns (`participant_id`, the feature names
above, and `outcome` set to `KNOW`, `TOT` or `DONT_KNOW`):

```bash
python train.py --data path/to/your_trials.csv
```

## Results (simulated data, 2,160 trials from held-out participants)

| Model | Accuracy | Macro F1 | TOT F1 | TOT-vs-rest ROC AUC |
|---|---|---|---|---|
| Majority class | 0.621 | — | — | — |
| Softmax regression (same engine, no hidden layer) | 0.848 | 0.798 | 0.675 | 0.911 |
| **Feedforward net (22 → 32 → 3)** | **0.851** | **0.805** | **0.686** | **0.917** |

Confusion matrix for the network (rows = true, columns = predicted):

```
             KNOW   TOT   DONT_KNOW
KNOW         1196   122      23
TOT            39   322      81
DONT_KNOW       3    53     321
```

The most important features by permutation importance are recall response time, word
frequency, first-letter access, feeling of knowing and recognition, which match the
measures TOT research treats as diagnostic.

**An honest take:** the network beats the majority baseline by 23 points, but it only
slightly beats a well-tuned linear model on this data, and only on AUC and F1. Most of the
signal here is roughly additive, so the extra capacity of a hidden layer adds little.
Comparing against a linear baseline built with the same engine is what makes that visible.
Real behavioral data with stronger interactions may show a bigger gap.

## Quick start

```bash
pip install -r requirements.txt
python train.py                  # simulate data, gradient-check, train, evaluate
python predict.py --demo         # score three example trials
python predict.py --input new_trials.csv --output predictions.csv
python -m pytest tests -q        # 10 tests
```

Useful options for `train.py`: `--hidden 64 32`, `--dropout 0.2`, `--lr 1e-3`, `--l2 1e-4`,
`--epochs 150`, `--no-class-weights`.

## Project structure

```
i-know-i-know-it/
├── totnet/
│   ├── layers.py      # Dense, activations, Dropout — forward + backward
│   ├── losses.py      # softmax cross-entropy, binary cross-entropy
│   ├── optim.py       # SGD (momentum), Adam
│   ├── network.py     # FeedforwardNet: backprop, fit, predict, save/load
│   ├── gradcheck.py   # finite-difference gradient verification
│   ├── metrics.py     # confusion matrix, P/R/F1, ROC AUC
│   └── data.py        # simulator, CSV loader, preprocessing, participant split
├── web/index.html     # interactive demo running the trained network in the browser
├── train.py           # end-to-end training and evaluation
├── predict.py         # inference on new trials
├── tests/test_totnet.py
├── data/tot_trials_simulated.csv
└── results/           # trained model, metrics.json, history.json, train_log.txt
```

## Tests

Ten tests cover:
- gradient checks across every layer type, both losses, L2 and class weights
- learning XOR, which proves the hidden layer captures non-linear patterns
- SGD convergence
- dropout behavior
- ROC AUC correctness
- save / load round-trip
- no participant leakage across splits
- the simulated bilingual TOT effect
