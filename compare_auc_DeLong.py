"""
Paired DeLong comparison of two AUCs on the same images, patient-clustered.

This is DeLong et al. (1988) used for its original purpose — comparing two
CORRELATED ROC curves — rather than testing one curve against chance. It was
written to answer: does a CXR image model discriminate insurance status better
than a model given demographics alone?

    python compare_auc_DeLong.py \\
        --set_a bootstrap_results_ml/*_predictions.csv \\
        --set_b bootstrap_results/exp0/MIMIC_*_predictions.csv

WHY PAIRED
----------
The two models are evaluated on the SAME images, so their AUCs are positively
correlated: an image that is easy for one tends to be easy for the other. An
unpaired comparison throws that correlation away and is badly underpowered.
The paired test forms the difference per image before taking any variance, so
the shared difficulty cancels.

Writing both models' influence expansions (see statistical_testing_DeLong.py):

    AUC_A - AUC_B  ~=  sum over images of [ contrib_A(i) - contrib_B(i) ]

so the difference has per-image contributions d(i) = contrib_A(i) - contrib_B(i),
and the SAME cluster machinery applies: sum d(i) within each patient, then take
the variance across patients.

    Var(AUC_A - AUC_B) = K/(K-1) * sum_k [ sum_{i in patient k} d(i) ]**2

This one expression handles BOTH sources of dependence at once — the pairing
(same images) and the clustering (repeated CXRs per patient). Neither is
assumed away.

LABEL POLARITY
--------------
The CNN pipeline (RawImageDataset.py) encodes Private as class 1, while the ML
notebook's LabelEncoder makes public class 1 — exactly inverted. Comparing them
naively would compare an AUC for "detects private" against one for "detects
public".

Because AUC(y, s) == AUC(1-y, 1-s), a file using the opposite convention is
realigned by flipping BOTH its labels and its scores; its own AUC is unchanged.
This script detects the mismatch by joining on sample_id and comparing the
true_label columns, reports what it found, and realigns everything to a single
stated convention. --positive_class chooses which convention that is.
Pass --no_auto_polarity to disable the realignment and fail loudly instead.

Only images present in BOTH files are used; the intersection is reported.
"""

import argparse
import csv
import glob
import math
import os

import numpy as np
from scipy import stats

from statistical_testing_DeLong import (
    benjamini_hochberg,
    cluster_var_from_contributions,
    delong_components,
    influence_contributions,
)


def load_rows(path):
    """sample_id -> (true_label, prob_1, patient_id) for one predictions CSV."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["sample_id"]] = (int(r["true_label"]),
                                   float(r["prob_1"]),
                                   r["patient_id"].strip())
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def align(a_rows, b_rows, auto_polarity=True):
    """Join two prediction files on sample_id and put them on one label convention.

    Returns (y, score_a, score_b, patients, n_common, flipped) where `y` is the
    label under file A's convention and score_b has been realigned if needed.
    """
    common = sorted(set(a_rows) & set(b_rows))
    if not common:
        raise ValueError("no shared sample_id between the two files")

    ya = np.array([a_rows[d][0] for d in common])
    yb = np.array([b_rows[d][0] for d in common])
    sa = np.array([a_rows[d][1] for d in common], dtype=float)
    sb = np.array([b_rows[d][1] for d in common], dtype=float)
    pats = np.array([a_rows[d][2] for d in common])
    pats_b = np.array([b_rows[d][2] for d in common])

    if not np.array_equal(pats, pats_b):
        n_bad = int((pats != pats_b).sum())
        raise ValueError(f"patient_id disagrees on {n_bad} shared images — "
                         f"the two files do not describe the same cohort")

    agree = float((ya == yb).mean())
    flipped = False
    if agree < 1.0:
        if agree > 0.0:
            raise ValueError(
                f"labels agree on only {agree:.1%} of shared images — neither the "
                f"same nor a clean inversion; refusing to guess a convention")
        if not auto_polarity:
            raise ValueError(
                "label conventions are inverted and --no_auto_polarity was set")
        # Perfect disagreement: B uses the opposite class as positive.
        # AUC(y, s) == AUC(1-y, 1-s), so flipping both leaves B's own AUC intact.
        sb = 1.0 - sb
        flipped = True

    return ya, sa, sb, pats, len(common), flipped


def paired_delong(y, score_a, score_b, clusters, alpha=0.05):
    """Patient-clustered paired DeLong test of H0: AUC_A == AUC_B."""
    auc_a, v10a, v01a, pm = delong_components(y, score_a)
    auc_b, v10b, v01b, _ = delong_components(y, score_b)

    ca = influence_contributions(auc_a, v10a, v01a, pm)
    cb = influence_contributions(auc_b, v10b, v01b, pm)

    var_d, k = cluster_var_from_contributions(ca - cb, clusters)
    # Each model's own clustered variance, for context and for the correlation
    var_a, _ = cluster_var_from_contributions(ca, clusters)
    var_b, _ = cluster_var_from_contributions(cb, clusters)

    diff = auc_a - auc_b
    se = math.sqrt(var_d) if var_d == var_d and var_d > 0 else float("nan")

    if se == se and se > 0:
        z = diff / se
        p = 2.0 * stats.norm.sf(abs(z))
    else:
        z, p = float("nan"), float("nan")

    # Correlation the pairing buys us: Var(A-B) = Var(A)+Var(B)-2*Cov
    if var_a > 0 and var_b > 0:
        cov = (var_a + var_b - var_d) / 2.0
        corr = cov / math.sqrt(var_a * var_b)
        se_unpaired = math.sqrt(var_a + var_b)
    else:
        corr, se_unpaired = float("nan"), float("nan")

    zc = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": diff,
        "se_diff": se, "se_diff_unpaired": se_unpaired, "corr": corr,
        "se_a": math.sqrt(var_a), "se_b": math.sqrt(var_b),
        "z": z, "p": p, "n_patients": k,
        "ci_lower": diff - zc * se, "ci_upper": diff + zc * se,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Patient-clustered paired DeLong comparison of two AUCs")
    parser.add_argument("--set_a", type=str, nargs="+", required=True,
                        help="Predictions CSV(s) for side A")
    parser.add_argument("--set_b", type=str, nargs="+", required=True,
                        help="Predictions CSV(s) for side B; every A-B pair is tested")
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--positive_class", type=str, default="file A's convention",
                        help="Name of the class treated as positive, for the report")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no_auto_polarity", action="store_true",
                        help="Fail instead of realigning inverted label conventions")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    a_paths = sorted({p for g in args.set_a for p in glob.glob(g)})
    b_paths = sorted({p for g in args.set_b for p in glob.glob(g)})
    if not a_paths or not b_paths:
        raise SystemExit("no files matched --set_a / --set_b")

    def short(p):
        return os.path.basename(p).replace("_predictions.csv", "")

    a_rows = {short(p): load_rows(p) for p in a_paths}
    b_rows = {short(p): load_rows(p) for p in b_paths}

    rows = []
    flipped_any = False
    for an, ar in a_rows.items():
        for bn, br in b_rows.items():
            try:
                y, sa, sb, pats, n_common, flipped = align(
                    ar, br, auto_polarity=not args.no_auto_polarity)
            except ValueError as e:
                print(f"  SKIP {an} vs {bn}: {e}")
                continue
            flipped_any |= flipped
            res = paired_delong(y, sa, sb, pats, alpha=args.alpha)
            res.update({"model_a": an, "model_b": bn,
                        "n_images": n_common, "polarity_flipped_b": flipped})
            rows.append(res)

    if not rows:
        raise SystemExit("no comparable pairs")

    qs = benjamini_hochberg([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q"] = q
        r["significant_fdr"] = bool(q < args.alpha)

    ex = rows[0]
    print(f"\n{'=' * 122}")
    print(f"Paired DeLong comparison — {args.label_a}  vs  {args.label_b}")
    print(f"Patient-clustered SE, same images, entire test set (no resampling)")
    print(f"{'=' * 122}")
    print(f"  Shared images : {ex['n_images']} from {ex['n_patients']} patients")
    print(f"  Positive class: {args.positive_class}")
    if flipped_any:
        print(f"  NOTE: side B used the opposite label convention; its labels and "
              f"scores were both flipped,")
        print(f"        which leaves each model's own AUC unchanged and puts both "
              f"on one convention.")
    print()
    hdr = (f"{'Model A':<16} {'Model B':<34} | {'AUC A':>6} | {'AUC B':>6} | "
           f"{'A - B':>7} | {'95% CI':>18} | {'SE':>6} | {'r':>5} | "
           f"{'p':>9} | {'q(BH)':>9} |")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["diff"]):
        mark = "  ***" if r["q"] < 0.001 else ("   **" if r["q"] < 0.01 else
               ("    *" if r["q"] < args.alpha else "   ns"))
        print(f"{r['model_a'][:16]:<16} {r['model_b'][:34]:<34} | "
              f"{r['auc_a']:>6.4f} | {r['auc_b']:>6.4f} | {r['diff']:>+7.4f} | "
              f"[{r['ci_lower']:>+.4f},{r['ci_upper']:>+.4f}] | {r['se_diff']:>6.4f} | "
              f"{r['corr']:>5.2f} | {r['p']:>9.3g} | {r['q']:>9.3g} |{mark}")

    n_sig = sum(r["significant_fdr"] for r in rows)
    print(f"\n  {n_sig}/{len(rows)} pairs differ at BH-FDR q < {args.alpha}.")
    print(f"  r = correlation between the two models' AUC influence functions; "
          f"the pairing is what makes")
    print(f"      the comparison sensitive — median SE(A-B) {np.median([r['se_diff'] for r in rows]):.4f} "
          f"vs {np.median([r['se_diff_unpaired'] for r in rows]):.4f} if the two were treated as independent.")
    print(f"{'=' * 122}")

    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fields = ["model_a", "model_b", "n_images", "n_patients",
                  "polarity_flipped_b", "auc_a", "auc_b", "diff", "se_diff",
                  "se_diff_unpaired", "corr", "se_a", "se_b", "z", "p", "q",
                  "ci_lower", "ci_upper", "significant_fdr"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
