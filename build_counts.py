"""
DDXPlus -> conditional probability tables.

Run ONCE. Commit the output. Never make the demo machine rebuild from 1.3M rows.

    python build_counts.py --data ./ddxplus --out counts.npz

Expects the DDXPlus release files in --data:
    release_train_patients.csv   (or .zip/.json — see --split)
    release_evidences.json       (code -> human-readable name, question text)
    release_conditions.json      (pathology metadata)

DDXPlus evidence encoding, which is the only fiddly part:
    "E_53"              binary, present
    "E_54_@_V_11"       categorical, took value V_11
    "E_55_@_4"          numeric,     took value 4

We flatten every (code, value) pair into its own binary feature:
    E_53                -> feature "E_53"
    E_54_@_V_11         -> feature "E_54=V_11"
So the model stays a clean Bernoulli Naive Bayes and you never special-case
value types at inference time.
"""
import argparse, ast, json, os, sys
from collections import Counter, defaultdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas numpy pyyaml")


def parse_evidences(raw):
    """'[\"E_53\", \"E_54_@_V_11\"]' -> ['E_53', 'E_54=V_11']"""
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    out = []
    for e in raw or []:
        e = str(e)
        if "_@_" in e:
            code, val = e.split("_@_", 1)
            out.append(f"{code}={val}")
        else:
            out.append(e)
    return out


def feature_universe(evidences_json):
    """
    Every feature that COULD be observed, from release_evidences.json.

    The universe must come from the SCHEMA, not from the training rows —
    otherwise a feature that never appears in training is invisible to the
    question engine, and "nobody asked" gets silently conflated with "that
    isn't a thing".

    Real numbers from the DDXPlus release: 223 evidence codes, of which 208
    are binary, 10 categorical and 5 multi-choice. Flattening each (code,
    value) pair into its own binary feature gives a universe of **972**, not
    223 — four location variables (E_55 pain site, E_57 radiation, E_133
    affected region, E_152 swelling site) carry 165 values each.

    That flattening is right for the likelihood but wrong for the question
    engine, which would otherwise propose "Do you have pain in the tonsil
    (L)?" as a yes/no question. So we also record each feature's PARENT code
    and the parent's real question text. `best_question` dedupes candidates
    by parent and renders the parent question ("Does the pain radiate to
    another location?") while keeping the winning value as the answer hint.
    """
    universe, questions, parents, parent_qs = [], {}, {}, {}
    for code, meta in evidences_json.items():
        name = meta.get("question_en") or meta.get("name") or code
        dtype = meta.get("data_type", "B")
        if dtype == "B":
            universe.append(code)
            questions[code] = name
            parents[code] = code
            parent_qs[code] = name
        else:
            for v in meta.get("possible-values", []):
                f = f"{code}={v}"
                universe.append(f)
                # value_meaning maps raw value -> {'fr': ..., 'en': ...}
                vm = (meta.get("value_meaning") or {}).get(str(v), {})
                label = vm.get("en", v)
                questions[f] = f"{name} [{label}]"
                parents[f] = code
                parent_qs[f] = name
    return universe, questions, parents, parent_qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./ddxplus")
    ap.add_argument("--out", default="counts.npz")
    ap.add_argument("--split", default="release_train_patients.csv")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    ap.add_argument("--limit", type=int, default=0, help="rows to read (0 = all)")
    ap.add_argument("--chunksize", type=int, default=100_000,
                    help="rows held in memory at once")
    args = ap.parse_args()

    ev_path = os.path.join(args.data, "release_evidences.json")
    with open(ev_path, encoding="utf-8") as f:
        evidences_json = json.load(f)
    universe, questions, parents, parent_qs = feature_universe(evidences_json)
    fidx = {f: i for i, f in enumerate(universe)}
    print(f"evidence codes: {len(evidences_json)}   flattened features: {len(universe)}")

    # Pathology list comes from the SCHEMA, not from whichever split we happen
    # to be counting. Otherwise a pathology absent from this split silently
    # vanishes from the model and the row indices shift between splits.
    cond_path = os.path.join(args.data, "release_conditions.json")
    with open(cond_path, encoding="utf-8") as f:
        paths = sorted(json.load(f).keys())
    pidx = {p: i for i, p in enumerate(paths)}
    print(f"pathologies: {len(paths)}")

    P, F = len(paths), len(universe)
    counts = np.zeros((P, F), dtype=np.float64)   # times feature seen w/ pathology
    totals = np.zeros(P, dtype=np.float64)        # patients per pathology
    unseen, unknown_paths = Counter(), Counter()

    # Streamed in chunks: the train split is ~1M rows and the EVIDENCES column
    # is a long string per row, so reading it whole costs several GB of RAM and
    # will thrash an 8GB laptop. Only two columns are ever needed.
    csv_path = os.path.join(args.data, args.split)
    print(f"reading {csv_path} in chunks of {args.chunksize:,} …")
    seen_rows = 0
    reader = pd.read_csv(csv_path, usecols=["PATHOLOGY", "EVIDENCES"],
                         chunksize=args.chunksize, nrows=args.limit or None)
    for chunk in reader:
        for pathology, evraw in zip(chunk["PATHOLOGY"].values,
                                    chunk["EVIDENCES"].values):
            pi = pidx.get(pathology)
            if pi is None:
                unknown_paths[pathology] += 1
                continue
            totals[pi] += 1
            for feat in parse_evidences(evraw):
                j = fidx.get(feat)
                if j is None:
                    unseen[feat] += 1
                    continue
                counts[pi, j] += 1
        seen_rows += len(chunk)
        print(f"  {seen_rows:,} rows", end="\r", flush=True)
    print(f"  {seen_rows:,} rows read")

    if unknown_paths:
        print(f"WARNING {len(unknown_paths)} pathologies in data but not in "
              f"release_conditions.json: {list(unknown_paths)[:5]}")
    empty = [p for p, t in zip(paths, totals) if t == 0]
    if empty:
        print(f"NOTE {len(empty)} pathologies have no examples in this split "
              f"(they fall back to the prior): {empty[:5]}")

    if unseen:
        print(f"WARNING {len(unseen)} features in data but not in schema, "
              f"e.g. {list(unseen)[:5]} — check your release_evidences.json")

    # P(feature=1 | pathology), Laplace-smoothed. Never 0 and never 1:
    # a hard zero makes one absent finding veto a pathology outright, which
    # is exactly the over-confidence a can't-miss list must not have.
    cond = (counts + args.alpha) / (totals[:, None] + 2 * args.alpha)
    prior = totals / totals.sum()

    np.savez_compressed(
        args.out,
        cond=cond, prior=prior,
        pathologies=np.array(paths, dtype=object),
        features=np.array(universe, dtype=object),
        questions=np.array([questions.get(f, f) for f in universe], dtype=object),
        parents=np.array([parents.get(f, f) for f in universe], dtype=object),
        parent_questions=np.array([parent_qs.get(f, f) for f in universe], dtype=object),
        totals=totals,
    )
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({P} pathologies x {F} features, {mb:.1f} MB)")
    print("COMMIT THIS FILE.")


if __name__ == "__main__":
    main()
