"""Data for tip-of-the-tongue (TOT) prediction.

Each row is one retrieval trial: a participant sees a definition or picture and
tries to name the target word in a given language. The outcome is one of

    0 = KNOW        the word was retrieved
    1 = TOT         "I know it, I know it" - the word feels known but won't come
    2 = DONT_KNOW   the word isn't known

Two sources are supported:

* `simulate_tot_dataset` - a synthetic multilingual dataset whose structure
  follows well-replicated TOT findings (see README). Use it to develop and
  demonstrate the model when no real data is available.
* `load_csv` - a real dataset in the same column format.

Prediction uses only process and item measures. The correctness of the recall
attempt itself is never a feature, which would make the task trivial.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CLASS_NAMES = ["KNOW", "TOT", "DONT_KNOW"]
LANGUAGES = ["English", "Spanish", "Catalan", "French", "Mandarin"]
ROLES = ["dominant", "nondominant", "third"]

NUMERIC_FEATURES = [
    "proficiency",            # self-rated proficiency in the test language, 0-1
    "word_log_freq",          # log10 frequency per million words
    "age_of_acquisition",     # years
    "n_syllables",
    "participant_age",
    "recall_rt_ms",           # time spent on the retrieval attempt
    "recognition_rt_ms",      # time to pick the target in a later recognition test
    "feeling_of_knowing",     # 1-6 rating given during the attempt
]
LOG_FEATURES = {"recall_rt_ms", "recognition_rt_ms"}   # log-transformed before scaling
BINARY_FEATURES = [
    "is_cognate",             # form overlaps with its translation in another known language
    "is_proper_name",
    "partial_first_letter",   # reported the target's first letter correctly
    "partial_syllables",      # guessed the syllable count correctly
    "translation_recalled",   # produced the word in another of their languages
    "recognition_correct",    # picked the target in the recognition test
]
CATEGORICAL_FEATURES = {"test_language_role": ROLES, "test_language": LANGUAGES}
ROMANCE_OR_ENGLISH = {"English", "Spanish", "Catalan", "French"}


# --------------------------------------------------------------------- simulate
def simulate_tot_dataset(n_participants: int = 160, trials_per_participant: int = 90,
                         seed: int = 42) -> list[dict]:
    """Generate synthetic multilingual TOT trials.

    Built-in effects, each based on the TOT literature:
      - lower word frequency and later age of acquisition -> more TOTs
      - testing in a non-dominant language -> more TOTs (bilingual TOT effect)
      - cognates -> fewer TOTs; proper names -> more TOTs
      - older adults -> more TOTs
      - TOTs come with long retrieval times, high feeling-of-knowing, frequent
        partial phonological access (first letter, syllables), and good later
        recognition; don't-know trials show the opposite pattern
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for pid in range(n_participants):
        langs = rng.permutation(LANGUAGES)
        n_langs = 3 if rng.random() < 0.35 else 2
        known = {"dominant": langs[0], "nondominant": langs[1], "third": langs[2]}
        prof = {"dominant": rng.uniform(0.85, 1.0),
                "nondominant": rng.uniform(0.35, 0.9),
                "third": rng.uniform(0.15, 0.6)}
        age = int(np.clip(rng.normal(38, 16), 18, 85))
        person_effect = rng.normal(0, 0.35)                 # individual retrieval skill
        speed = rng.lognormal(0, 0.25) * (1 + 0.006 * (age - 38))   # general response speed
        fok_bias = rng.normal(0, 0.7)                       # tendency to rate high or low
        persistence = rng.uniform(0.4, 1.6)                 # how long they keep searching
        roles = ROLES[:n_langs]
        role_p = [0.45, 0.4, 0.15][:n_langs]
        role_p = np.array(role_p) / sum(role_p)

        for _ in range(trials_per_participant):
            role = roles[rng.choice(len(roles), p=role_p)]
            lang = known[role]
            p = prof[role]
            freq = rng.uniform(0.3, 5.0)
            aoa = float(np.clip(rng.normal(9.5 - 1.1 * freq, 2.5), 2, 20))
            syll = int(np.clip(rng.poisson(1.8) + 1, 1, 6))
            cognate_p = 0.45 if lang in ROMANCE_OR_ENGLISH else 0.05
            cognate = rng.random() < cognate_p
            proper = rng.random() < 0.15

            # 1) is the concept/word known at all? (semantic + lexical knowledge)
            know_logit = -1.0 + 0.9 * freq + 2.0 * p + 0.6 * cognate - 0.08 * aoa
            word_known = rng.random() < 1 / (1 + np.exp(-know_logit))

            # 2) if known, is its sound form accessible right now?
            access = (0.2 + 0.55 * (freq - 2.6) + 2.2 * (p - 0.7) + 0.7 * cognate
                      - 0.9 * proper - 0.06 * (aoa - 7) - 0.12 * (syll - 2)
                      - 0.018 * (age - 38) + person_effect + rng.normal(0, 0.75))
            if not word_known:
                outcome = 2
            elif access > -0.15:
                outcome = 0
            elif access > -1.6:
                outcome = 1
            else:
                outcome = 2   # weak access feels like not knowing

            # 3) observable process measures, conditional on the hidden state
            #    (people differ in speed, rating style and persistence, so the
            #    same state looks different across participants)
            fok_core = 3.4 + 1.2 * np.tanh(access) if word_known else 2.2
            fok = fok_core + 0.6 * (outcome == 1) + fok_bias + rng.normal(0, 1.2)
            if outcome == 0:
                rt = speed * rng.lognormal(np.log(1900) - 0.2 * access, 0.5)
                first, sylls = rng.random() < 0.8, rng.random() < 0.65
                rec = rng.random() < 0.93
                rec_rt = speed * rng.lognormal(np.log(1300), 0.4)
                trans_p = 0.75
            elif outcome == 1:
                # people persist longer when the word feels closer
                rt = speed * rng.lognormal(np.log(1800 + 3200 * persistence)
                                           + 0.15 * (fok - 4), 0.55)
                pa = 1 / (1 + np.exp(-(1.2 * access + 1.0)))
                first, sylls = rng.random() < 0.25 + 0.4 * pa, rng.random() < 0.3 + 0.25 * pa
                rec = rng.random() < 0.8
                rec_rt = speed * rng.lognormal(np.log(1450), 0.4)
                trans_p = 0.5 + 0.25 * (role != "dominant")     # word known in L1, stuck in L2
            else:
                rt = speed * rng.lognormal(np.log(1200 + 2000 * persistence), 0.55)
                first, sylls = rng.random() < 0.15, rng.random() < 0.27
                rec = rng.random() < (0.55 if word_known else 0.33)
                rec_rt = speed * rng.lognormal(np.log(1800), 0.4)
                trans_p = 0.3
            rows.append({
                "participant_id": pid, "test_language": lang, "test_language_role": role,
                "proficiency": round(float(p), 3), "word_log_freq": round(float(freq), 3),
                "age_of_acquisition": round(aoa, 2), "n_syllables": syll,
                "participant_age": age, "recall_rt_ms": round(float(rt)),
                "recognition_rt_ms": round(float(rec_rt)),
                "feeling_of_knowing": int(np.clip(round(fok), 1, 6)),
                "is_cognate": int(cognate), "is_proper_name": int(proper),
                "partial_first_letter": int(first), "partial_syllables": int(sylls),
                "translation_recalled": int(rng.random() < trans_p),
                "recognition_correct": int(rec), "outcome": CLASS_NAMES[outcome],
            })
    return rows


# --------------------------------------------------------------------- csv io
def save_csv(rows: list[dict], path: str | Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = set(NUMERIC_FEATURES + BINARY_FEATURES + list(CATEGORICAL_FEATURES)
                   + ["participant_id", "outcome"])
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return rows


# --------------------------------------------------------------- preprocessing
class Preprocessor:
    """Turns trial dicts into a numeric matrix: log-transform RTs, z-score
    numeric columns (statistics fit on training data only), pass binaries
    through, and one-hot encode categoricals."""

    def fit(self, rows: list[dict]) -> "Preprocessor":
        num = self._numeric(rows)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0) + 1e-8
        return self

    def _numeric(self, rows):
        cols = []
        for name in NUMERIC_FEATURES:
            v = np.array([float(r[name]) for r in rows])
            cols.append(np.log(v) if name in LOG_FEATURES else v)
        return np.column_stack(cols)

    def transform(self, rows: list[dict]) -> np.ndarray:
        num = (self._numeric(rows) - self.mean) / self.std
        binary = np.array([[float(r[b]) for b in BINARY_FEATURES] for r in rows])
        onehots = []
        for col, cats in CATEGORICAL_FEATURES.items():
            oh = np.zeros((len(rows), len(cats)))
            for i, r in enumerate(rows):
                if r[col] in cats:
                    oh[i, cats.index(r[col])] = 1.0
            onehots.append(oh)
        return np.hstack([num, binary] + onehots)

    @property
    def feature_names(self) -> list[str]:
        names = list(NUMERIC_FEATURES) + list(BINARY_FEATURES)
        for col, cats in CATEGORICAL_FEATURES.items():
            names += [f"{col}={c}" for c in cats]
        return names

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "Preprocessor":
        p = cls()
        p.mean, p.std = np.array(state["mean"]), np.array(state["std"])
        return p


def encode_labels(rows: list[dict]) -> np.ndarray:
    return np.array([CLASS_NAMES.index(r["outcome"]) for r in rows])


def split_by_participant(rows: list[dict], val_frac: float = 0.15, test_frac: float = 0.15,
                         seed: int = 0):
    """Split so each participant appears in exactly one of train/val/test,
    which measures generalization to new people rather than memorized ones."""
    pids = np.array(sorted({int(r["participant_id"]) for r in rows}))
    np.random.default_rng(seed).shuffle(pids)
    n_test = int(len(pids) * test_frac)
    n_val = int(len(pids) * val_frac)
    test_ids, val_ids = set(pids[:n_test]), set(pids[n_test:n_test + n_val])
    train, val, test = [], [], []
    for r in rows:
        pid = int(r["participant_id"])
        (test if pid in test_ids else val if pid in val_ids else train).append(r)
    return train, val, test
