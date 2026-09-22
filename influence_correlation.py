"""
Influence-function correlation between models evaluated on the same patients.

For each model, every patient k gets one influence term h_k -- how much that
patient pulled the model's AUC up or down (see statistical_testing_DeLong.py for
the derivation). Two models scored on the same patients give two vectors of h_k,
and their correlation r answers:

    when model A does unusually well on a given patient, does B as well?

Why r is worth reporting
------------------------
1. It is what determines whether a paired AUC comparison can resolve anything:

       Var(AUC_A - AUC_B) = Var(A) + Var(B) - 2*r*sqrt(Var_A * Var_B)

   At r near 1 the shared noise cancels and tiny differences become detectable;
   at r near 0 nothing cancels and the comparison is underpowered. Two gaps of
   the same size can therefore land on opposite sides of p = 0.05 purely because
   of r, which is easy to misread as a difference in effect size.

2. It is a substantive result in its own right. Low r between two models with
   similar AUC means they achieve that discrimination on DIFFERENT patients --
   i.e. they carry partly independent information and are complementary rather
   than redundant.

The script reports r two ways as a self-check: computed directly from the
per-patient h_k vectors, and recovered from the variance identity
Cov = (Var_A + Var_B - Var_{A-B}) / 2. They are algebraically the same quantity,
so a mismatch means something is wrong.

Label polarity
--------------
The CNN pipeline (RawImageDataset.py) encodes Private as class 1 while the ML
notebooks make public class 1. Files are aligned to the first model's convention
by flipping both labels and scores where needed, which leaves each model's own
AUC unchanged. Detected automatically and reported.

Usage:
    # The three pairings, using the defaults below
    python3 influence_correlation.py

    # Any set of models
    python3 influence_correlation.py \\
        --model "Demographics=bootstrap_results_ml/XGBoost_predictions.csv" \\
        --model "Image only=bootstrap_results/exp0/MIMIC_densenet_exp0_densenet_Rand123_predictions.csv" \\
        --model "Image+demo=bootstrap_results/exp3/MIMIC_medgemma_densenet_sexagerace_Rand123_predictions.csv" \\
        --output influence_correlation.csv
"""

import argparse
import csv
import itertools
import math
import os

import numpy as np

from statistical_testing_DeLong import (
    delong_components,
    influence_contributions,
)

# Defaults: the demographics-only model, the image-only CNN, and the nested
# image+demographics CNN (same demographics as the ML model, plus the image).
DEFAULT_MODELS = [
    ("Demographics",
     "bootstrap_results_ml/XGBoost_predictions.csv"),
    ("Image only",
     "bootstrap_results/exp0/MIMIC_densenet_exp0_densenet_Rand123_predictions.csv"),
    ("Image+demo",
     "bootstrap_results/exp3/MIMIC_medgemma_densenet_sexagerace_Rand123_predictions.csv"),
]


def load_rows(path, label_col="true_label", score_col="prob_1",
              id_col="sample_id", cluster_col="patient_id"):
    """sample_id -> (label, score, patient_id) for one predictions CSV."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in (id_col, label_col, score_col, cluster_col):
            if col not in reader.fieldnames:
                raise ValueError(f"{path}: missing column '{col}'")
        for r in reader:
            out[r[id_col]] = (int(r[label_col]), float(r[score_col]),
                              r[cluster_col].strip())
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def align_models(models):
    """Join every model on the COMMON sample_id set and unify label polarity.

    Using one intersection across all models (rather than a different one per
    pair) keeps every h_k vector defined on the same patients, so the pairwise
    correlations are mutually comparable.

    Returns (ids, patients, y, {name: scores}, flipped_names).
    """
    common = set.intersection(*(set(rows) for _, rows in models))
    if not common:
        raise ValueError("no sample_id shared by all models")
    ids = sorted(common)

    ref_name, ref_rows = models[0]
    y = np.array([ref_rows[i][0] for i in ids])
    patients = np.array([ref_rows[i][2] for i in ids])

    scores, flipped = {}, []
    for name, rows in models:
        lab = np.array([rows[i][0] for i in ids])
        sc = np.array([rows[i][1] for i in ids], dtype=float)
        pat = np.array([rows[i][2] for i in ids])

        if not np.array_equal(pat, patients):
            n_bad = int((pat != patients).sum())
            raise ValueError(f"{name}: patient_id disagrees with "
                             f"{ref_name} on {n_bad} shared images")

        agree = float((lab == y).mean())
        if agree == 1.0:
            pass
        elif agree == 0.0:
            # Opposite convention. AUC(y, s) == AUC(1-y, 1-s), so flipping both
            # realigns the model without changing its own AUC.
            sc = 1.0 - sc
            flipped.append(name)
        else:
            raise ValueError(
                f"{name}: labels agree with {ref_name} on only {agree:.1%} of "
                f"shared images -- neither the same convention nor a clean "
                f"inversion; refusing to guess")
        scores[name] = sc
    return ids, patients, y, scores, flipped


def per_patient_influence(y, score, patients):
    """One influence term h_k per patient, plus the model's AUC."""
    auc, v10, v01, pos_mask = delong_components(y, score)
    contrib = influence_contributions(auc, v10, v01, pos_mask)
    uniq, inv = np.unique(patients, return_inverse=True)
    h = np.zeros(len(uniq), dtype=float)
    np.add.at(h, inv, contrib)
    return auc, h


def var_from_h(h):
    """Cluster-robust variance of an AUC from its per-patient influence terms.

    Same estimator as cluster_var_from_contributions in statistical_testing_DeLong,
    applied to terms that are already summed within patient. The h_k sum to zero
    by construction, so no centering is needed.
    """
    k = len(h)
    return (k / (k - 1.0)) * float(np.sum(h ** 2))


def main():
    ap = argparse.ArgumentParser(
        description="Influence-function correlation between models on the same patients")
    ap.add_argument("--model", action="append", default=None, metavar="NAME=PATH",
                    help="Labelled predictions CSV; repeatable. "
                         "Defaults to the demographics / image-only / image+demo trio.")
    ap.add_argument("--score_col", default="prob_1")
    ap.add_argument("--label_col", default="true_label")
    ap.add_argument("--output", default=None, help="Optional CSV of the pairwise results")
    args = ap.parse_args()

    if args.model:
        spec = []
        for m in args.model:
            if "=" not in m:
                raise SystemExit(f"--model needs NAME=PATH, got {m!r}")
            name, path = m.split("=", 1)
            spec.append((name.strip(), path.strip()))
    else:
        spec = DEFAULT_MODELS

    missing = [p for _, p in spec if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing prediction file(s):\n  " + "\n  ".join(missing))

    models = [(name, load_rows(path, args.label_col, args.score_col))
              for name, path in spec]
    ids, patients, y, scores, flipped = align_models(models)

    names = [n for n, _ in spec]
    auc, H = {}, {}
    for n in names:
        auc[n], H[n] = per_patient_influence(y, scores[n], patients)

    n_pat = len(np.unique(patients))
    print(f"\n{'=' * 78}")
    print("Influence-function correlation")
    print(f"{'=' * 78}")
    print(f"  Shared images : {len(ids)} from {n_pat} patients")
    if flipped:
        print(f"  Polarity flipped to match '{names[0]}': {', '.join(flipped)}")
    print()
    print(f"  {'model':<16}{'AUC':>8}{'clustered SE':>15}   file")
    print("  " + "-" * 74)
    for n, (_, path) in zip(names, spec):
        print(f"  {n:<16}{auc[n]:>8.4f}{math.sqrt(var_from_h(H[n])):>15.4f}"
              f"   {os.path.basename(path)[:38]}")

    print(f"\n  {'pair':<34}{'r direct':>10}{'r from Var':>12}"
          f"{'SE(A-B)':>10}{'SE if indep':>13}")
    print("  " + "-" * 79)

    rows = []
    for a, b in itertools.combinations(names, 2):
        r_direct = float(np.corrcoef(H[a], H[b])[0, 1])

        var_a, var_b = var_from_h(H[a]), var_from_h(H[b])
        var_d = var_from_h(H[a] - H[b])
        # Cov = (Var_A + Var_B - Var_{A-B}) / 2 -- the same r by a different route
        r_var = ((var_a + var_b - var_d) / 2.0) / math.sqrt(var_a * var_b)

        se_d = math.sqrt(var_d)
        se_indep = math.sqrt(var_a + var_b)
        print(f"  {a + '  vs  ' + b:<34}{r_direct:>10.4f}{r_var:>12.4f}"
              f"{se_d:>10.4f}{se_indep:>13.4f}")
        rows.append({
            "model_a": a, "model_b": b,
            "auc_a": auc[a], "auc_b": auc[b], "diff": auc[a] - auc[b],
            "r_direct": r_direct, "r_from_variance": r_var,
            "se_a": math.sqrt(var_a), "se_b": math.sqrt(var_b),
            "se_diff_paired": se_d, "se_diff_if_independent": se_indep,
            "variance_reduction_from_pairing": 1 - se_d / se_indep,
            "n_images": len(ids), "n_patients": n_pat,
        })

    worst = max(abs(r["r_direct"] - r["r_from_variance"]) for r in rows)
    print(f"\n  max |r_direct - r_from_Var| = {worst:.2e}  "
          f"{'OK (same quantity)' if worst < 1e-9 else 'MISMATCH'}")
    print(f"\n  r near 1 -> shared noise cancels, small AUC gaps are detectable.")
    print(f"  r near 0 -> nothing cancels; the pair is underpowered, and similar")
    print(f"              AUC is being reached on different patients.")
    print(f"{'=' * 78}")

    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
