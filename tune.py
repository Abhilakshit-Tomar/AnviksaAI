"""
Pick `tau` on the VALIDATE split. Never on test.

    python tune.py --data ./ddxplus --n 4000

Naive Bayes over a million rows produces conditionals sharp enough that
multiplying ten of them pins the posterior at 100.0% / 0.0%. `tau` damps each
factor (`logP += tau * log P(e|path)`), trading likelihood against prior.

**Choose tau by CALIBRATION, not accuracy.** Accuracy keeps improving as tau
rises and the model gets more confident, but this product lives or dies on the
5-20% band: Can't-Miss exists to surface a pulmonary embolism sitting at 8%.
A model that is 94% accurate and always says 100% is useless here, because
every posterior collapses and there is nothing left to rank. Pick the tau with
the best ECE among those whose top-5 recall is still near its maximum, and say
exactly that when a judge asks how you set it.

Reported per tau:
    top1 / top5   recall of the true pathology
    brier         mean squared error on the true-class probability (lower better)
    ECE           expected calibration error — |confidence - accuracy|, binned
    mean-max      average top probability. If this is ~1.00 the model is
                  saturated and the Can't-Miss panel will be empty.
"""
import argparse, ast, json, os

import numpy as np
import pandas as pd

from engine import Engine
from build_counts import parse_evidences


def ece(conf, correct, bins=10):
    """Expected calibration error over equal-width confidence bins."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./ddxplus")
    ap.add_argument("--split", default="release_validate_patients.csv")
    ap.add_argument("--counts", default="counts.npz")
    ap.add_argument("--severity", default="severity.yaml")
    ap.add_argument("--n", type=int, default=4000, help="patients to evaluate")
    ap.add_argument("--taus", default="1.0,0.5,0.3,0.2,0.15,0.12,0.1,0.07,0.05")
    args = ap.parse_args()

    eng = Engine.load(args.counts, args.severity)
    pidx = {p: i for i, p in enumerate(eng.pathologies)}

    df = pd.read_csv(os.path.join(args.data, args.split),
                     usecols=["PATHOLOGY", "EVIDENCES"], nrows=args.n)
    cases = []
    for path, evraw in zip(df["PATHOLOGY"].values, df["EVIDENCES"].values):
        if path not in pidx:
            continue
        cases.append(({f: True for f in parse_evidences(evraw)}, pidx[path]))
    print(f"{len(cases):,} validate cases\n")

    print(f"{'tau':>6} {'top1':>7} {'top5':>7} {'brier':>8} {'ECE':>7} {'mean-max':>9}")
    print("-" * 48)
    rows = []
    for tau in [float(t) for t in args.taus.split(",")]:
        eng.tau = tau
        t1 = t5 = 0
        brier, conf, corr = [], [], []
        for evidence, truth in cases:
            post = eng.posterior(evidence)
            order = np.argsort(-post)
            t1 += order[0] == truth
            t5 += truth in order[:5]
            brier.append((1.0 - post[truth]) ** 2)
            conf.append(post[order[0]])
            corr.append(order[0] == truth)
        n = len(cases)
        row = (tau, t1 / n, t5 / n, float(np.mean(brier)),
               ece(conf, corr), float(np.mean(conf)))
        rows.append(row)
        print(f"{row[0]:>6.2f} {row[1]:>7.1%} {row[2]:>7.1%} "
              f"{row[3]:>8.4f} {row[4]:>7.4f} {row[5]:>9.3f}")

    best_t5 = max(r[2] for r in rows)
    ok = [r for r in rows if r[2] >= best_t5 - 0.02]      # within 2pt of best top-5
    pick = min(ok, key=lambda r: r[4])                     # then best calibrated
    print(f"\nRecommended tau = {pick[0]}  "
          f"(top5 {pick[2]:.1%}, ECE {pick[4]:.4f}, mean-max {pick[5]:.2f})")
    print("Set it in engine.py, and report your final numbers on TEST, not this split.")
    if pick[5] > 0.95:
        print("WARNING mean-max is near 1.0 — the posterior is still saturated "
              "and Can't-Miss will have nothing to rank. Try smaller taus.")


if __name__ == "__main__":
    main()
