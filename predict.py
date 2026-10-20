"""Predict tip-of-the-tongue states for new trials with a trained model.

Examples:
    python predict.py --input data/new_trials.csv --output predictions.csv
    python predict.py --demo        # score a few hand-written example trials
"""
import argparse
import csv
import json
from pathlib import Path

from totnet import CLASS_NAMES, FeedforwardNet, Preprocessor, load_csv

DEMO_TRIALS = [
    # fast, confident retrieval of a common word in the dominant language
    {"test_language": "Spanish", "test_language_role": "dominant", "proficiency": 0.97,
     "word_log_freq": 4.2, "age_of_acquisition": 4, "n_syllables": 2, "participant_age": 25,
     "recall_rt_ms": 1400, "recognition_rt_ms": 1100, "feeling_of_knowing": 6, "is_cognate": 0,
     "is_proper_name": 0, "partial_first_letter": 1, "partial_syllables": 1,
     "translation_recalled": 1, "recognition_correct": 1},
    # rare word in a second language: long search, strong feeling of knowing,
    # first letter available, known in the other language -> classic TOT
    {"test_language": "English", "test_language_role": "nondominant", "proficiency": 0.6,
     "word_log_freq": 1.1, "age_of_acquisition": 14, "n_syllables": 4, "participant_age": 61,
     "recall_rt_ms": 8200, "recognition_rt_ms": 1400, "feeling_of_knowing": 5, "is_cognate": 0,
     "is_proper_name": 0, "partial_first_letter": 1, "partial_syllables": 0,
     "translation_recalled": 1, "recognition_correct": 1},
    # unfamiliar word in a weak third language
    {"test_language": "French", "test_language_role": "third", "proficiency": 0.25,
     "word_log_freq": 0.6, "age_of_acquisition": 17, "n_syllables": 3, "participant_age": 30,
     "recall_rt_ms": 2600, "recognition_rt_ms": 2300, "feeling_of_knowing": 1, "is_cognate": 0,
     "is_proper_name": 0, "partial_first_letter": 0, "partial_syllables": 0,
     "translation_recalled": 0, "recognition_correct": 0},
]


def main():
    p = argparse.ArgumentParser(description="Predict TOT states")
    p.add_argument("--model-dir", default="results")
    p.add_argument("--input", help="CSV of trials (outcome column optional)")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    model_dir = Path(a.model_dir)
    net = FeedforwardNet.load(model_dir / "tot_model")
    prep = Preprocessor.from_state(json.loads((model_dir / "preprocessor.json").read_text()))

    if a.demo:
        rows = DEMO_TRIALS
    elif a.input:
        with open(a.input, newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        p.error("pass --input FILE or --demo")

    proba = net.predict_proba(prep.transform(rows))
    for row, pr in zip(rows, proba):
        row["predicted"] = CLASS_NAMES[pr.argmax()]
        for name, v in zip(CLASS_NAMES, pr):
            row[f"p_{name}"] = round(float(v), 4)

    if a.demo:
        for i, row in enumerate(rows, 1):
            probs = "  ".join(f"{n} {row[f'p_{n}']:.2f}" for n in CLASS_NAMES)
            print(f"trial {i}: predicted {row['predicted']:10s} | {probs}")
    else:
        with open(a.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} predictions to {a.output}")


if __name__ == "__main__":
    main()
