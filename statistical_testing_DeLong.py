"""
DeLong test of AUC against the chance baseline (AUC = 0.5), cluster-adjusted
for patients contributing more than one chest X-ray.

Reads the per-sample `*_predictions.csv` files written by bootstrap_evaluate.py
(columns: patient_id, true_label, prob_1, ...), computes the AUC on the ENTIRE
test set — no resampling — and tests H0: AUC = 0.5 with a standard error that
accounts for the within-patient correlation between repeated CXRs.

WHY A CLUSTER ADJUSTMENT IS NEEDED
----------------------------------
The classical DeLong et al. (1988) variance treats every image as an
independent observation. In this test set one patient can contribute up to 15
CXRs, and — because insurance type is a patient-level attribute — every image
of a patient carries the SAME label. Repeated images of one patient are
therefore strongly positively correlated, and the number of independent units
is the number of patients (~1224), not the number of images (~1965). Using the
unadjusted variance understates the standard error, inflates |Z|, and produces
anti-conservative p-values.

METHOD
------
The AUC is a two-sample U-statistic and is asymptotically linear:

    AUC_hat - AUC  ~=  (1/m) * sum_i [V10(i) - AUC]  +  (1/n) * sum_j [V01(j) - AUC]

where the DeLong "structural components" (placement values) are

    V10(i) = proportion of negatives that positive i outranks   (ties count 1/2)
    V01(j) = proportion of positives that outrank negative j    (ties count 1/2)

and m, n are the numbers of positive and negative images. Taking the variance
of the right-hand side under independence reproduces DeLong exactly:

    Var_naive = S10/m + S01/n          (S10, S01 = sample variances of V10, V01)

To adjust for clustering we instead sum each patient's contribution to that
linear expansion into a single term and take the variance ACROSS PATIENTS —
the cluster-robust ("sandwich") form of the same estimator:

    h_k = (1/m) * sum_{i in patient k, positive} [V10(i) - AUC_hat]
        + (1/n) * sum_{j in patient k, negative} [V01(j) - AUC_hat]

    Var_cluster = K/(K-1) * sum_k h_k**2          (K = number of patients)

The h_k sum to exactly zero by construction, so no centering is needed. This is
the DeLong/U-statistic variance generalised to clustered ROC data in the sense
of Obuchowski (1997), Biometrics 53:567-578, "Nonparametric analysis of
clustered ROC curve data". It reduces to the classical DeLong variance when
every patient contributes exactly one image (verified by --self_test), and it
makes no assumption that a patient's images are exchangeable or equally
numerous.

The test statistic is then the usual

    Z = (AUC_hat - 0.5) / sqrt(Var_cluster),   two-sided p = 2 * (1 - Phi(|Z|))

Structural components are computed with the O(N log N) midrank algorithm of
Sun & Xu (2014), IEEE SPL 21:1389-1393, which handles tied scores exactly via
the 1/2 convention.

MULTIPLICITY
------------
exp1-exp5 together contain ~157 model/condition combinations. Testing each
against chance without correction would be expected to yield ~8 false
positives at alpha=0.05, so Benjamini-Hochberg FDR q-values and Bonferroni
p-values are reported alongside the raw p-values. Correction is applied within
each experiment and across all tests in the run (both are reported).

Usage:
    # All experiments found under bootstrap_results/
    python statistical_testing_DeLong.py \\
        --results_dir bootstrap_results \\
        --experiments exp1 exp2 exp3 exp4 exp5 \\
        --output "bootstrap_results/statistical tests/delong_vs_chance.csv"

    # A single predictions file
    python statistical_testing_DeLong.py \\
        --predictions bootstrap_results/exp1/MIMIC_densenet_keep_patch5_Rand123_predictions.csv

    # Show what ignoring the clustering would have given
    python statistical_testing_DeLong.py --results_dir bootstrap_results --show_naive

    # Verify the clustered estimator reduces to classical DeLong
    python statistical_testing_DeLong.py --self_test
"""

import argparse
import csv
import glob
import math
import os

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# DeLong structural components
# ──────────────────────────────────────────────────────────────────────────────

def _midrank(x):
    """Midranks of x (1-based); tied values share the average of their ranks."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        # mean of the 1-based ranks i+1 .. j
        ranks_sorted[i:j] = 0.5 * (i + j + 1)
        i = j

    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def delong_components(y_true, scores):
    """AUC plus the DeLong structural components V10 (positives) and V01 (negatives).

    Ties in `scores` are handled with the standard 1/2 convention.

    Returns
    -------
    auc      : float, the Mann-Whitney AUC on all supplied samples
    v10      : array of length m, one placement value per positive image
    v01      : array of length n, one placement value per negative image
    pos_mask : boolean array marking which input rows are positives
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    pos_mask = y_true == 1
    x = scores[pos_mask]      # positives
    y = scores[~pos_mask]     # negatives
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        raise ValueError(f"AUC undefined: {m} positives and {n} negatives")

    tx = _midrank(x)
    ty = _midrank(y)
    tz = _midrank(np.concatenate([x, y]))
    tz_x, tz_y = tz[:m], tz[m:]

    # (tz_x - tx) = negatives below each positive (+ half the ties)
    v10 = (tz_x - tx) / n
    # (tz_y - ty) = positives below each negative (+ half the ties)
    v01 = 1.0 - (tz_y - ty) / m

    auc = tz_x.sum() / (m * n) - (m + 1.0) / (2.0 * n)
    return auc, v10, v01, pos_mask


def delong_var_naive(auc, v10, v01):
    """Classical DeLong variance — assumes every image is independent."""
    m, n = len(v10), len(v01)
    if m < 2 or n < 2:
        return float("nan")
    s10 = v10.var(ddof=1)
    s01 = v01.var(ddof=1)
    return s10 / m + s01 / n


def influence_contributions(auc, v10, v01, pos_mask):
    """Per-image contributions to the AUC influence function, in original row order.

    AUC_hat - AUC ~= sum over images of these terms, so they are the building
    block for every variance below — single-model or paired.
    """
    m, n = len(v10), len(v01)
    contrib = np.empty(len(pos_mask), dtype=float)
    contrib[pos_mask] = (v10 - auc) / m
    contrib[~pos_mask] = (v01 - auc) / n
    return contrib


def cluster_var_from_contributions(contrib, clusters):
    """Cluster-robust variance of a sum of per-image influence contributions.

    Sums each patient's contributions into one term h_k and takes the variance
    across patients. Shared by the single-model and paired (two-model) tests.
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    h = np.zeros(len(uniq), dtype=float)
    np.add.at(h, inv, contrib)

    k = len(uniq)
    if k < 2:
        return float("nan"), k
    return (k / (k - 1.0)) * np.sum(h ** 2), k


def delong_var_clustered(auc, v10, v01, pos_mask, clusters):
    """Cluster-robust DeLong variance; the resampling/independence unit is the patient.

    Each patient's contributions to the AUC influence function are summed into a
    single term h_k, and the variance is taken across patients.
    """
    contrib = influence_contributions(auc, v10, v01, pos_mask)
    return cluster_var_from_contributions(contrib, clusters)


# ──────────────────────────────────────────────────────────────────────────────
# Test against the chance baseline
# ──────────────────────────────────────────────────────────────────────────────

def test_auc_vs_chance(y_true, scores, clusters=None, baseline=0.5, alpha=0.05):
    """DeLong test of H0: AUC = `baseline`, cluster-adjusted when `clusters` is given.

    Returns a dict of point estimate, standard errors, Z, p and confidence intervals.
    """
    auc, v10, v01, pos_mask = delong_components(y_true, scores)
    m, n = len(v10), len(v01)

    var_naive = delong_var_naive(auc, v10, v01)
    if clusters is not None:
        var_used, n_clusters = delong_var_clustered(auc, v10, v01, pos_mask, clusters)
        unit = "patient"
    else:
        var_used, n_clusters = var_naive, float("nan")
        unit = "image"

    se = math.sqrt(var_used) if var_used == var_used and var_used > 0 else float("nan")
    se_naive = (math.sqrt(var_naive)
                if var_naive == var_naive and var_naive > 0 else float("nan"))

    if se == se and se > 0:
        z = (auc - baseline) / se
        p = 2.0 * stats.norm.sf(abs(z))
    else:
        z, p = float("nan"), float("nan")

    zc = stats.norm.ppf(1 - alpha / 2)
    ci_lo, ci_hi = auc - zc * se, auc + zc * se

    # Logit-transformed interval: stays inside (0, 1) and has better small-sample
    # coverage than the Wald interval when the AUC sits near a boundary.
    if 0 < auc < 1 and se == se:
        logit = math.log(auc / (1 - auc))
        half = zc * se / (auc * (1 - auc))
        lo_l = 1 / (1 + math.exp(-(logit - half)))
        hi_l = 1 / (1 + math.exp(-(logit + half)))
    else:
        lo_l, hi_l = ci_lo, ci_hi

    # How much the clustering costs: variance ratio and the implied effective N
    if se == se and se_naive == se_naive and se_naive > 0:
        design_effect = var_used / var_naive
        n_eff = (m + n) / design_effect if design_effect > 0 else float("nan")
    else:
        design_effect, n_eff = float("nan"), float("nan")

    return {
        "auc": auc,
        "n_images": m + n,
        "n_pos": m,
        "n_neg": n,
        "n_patients": n_clusters,
        "unit": unit,
        "se": se,
        "se_naive": se_naive,
        "design_effect": design_effect,
        "n_effective": n_eff,
        "z": z,
        "p": p,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "ci_lower_logit": lo_l,
        "ci_upper_logit": hi_l,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multiplicity control
# ──────────────────────────────────────────────────────────────────────────────

def benjamini_hochberg(pvals):
    """BH step-up FDR q-values, aligned with the input order. NaNs stay NaN."""
    p = np.asarray(pvals, dtype=float)
    valid = np.where(~np.isnan(p))[0]
    q = np.full(len(p), float("nan"))
    if len(valid) == 0:
        return q

    sub = p[valid]
    order = np.argsort(sub)
    ranked = sub[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    q[valid] = out
    return q


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_predictions(path, label_col="true_label", score_col="prob_1",
                     cluster_col="patient_id"):
    """Read one *_predictions.csv; returns (labels, scores, clusters|None)."""
    labels, scores, clusters = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in (label_col, score_col):
            if col not in reader.fieldnames:
                raise ValueError(f"{path}: missing required column '{col}'")
        has_cluster = cluster_col in reader.fieldnames
        for row in reader:
            labels.append(int(row[label_col]))
            scores.append(float(row[score_col]))
            if has_cluster:
                clusters.append(row[cluster_col].strip())

    labels = np.array(labels)
    scores = np.array(scores, dtype=float)

    # Fall back to image-level inference rather than inventing patient ids
    if not has_cluster or any(c == "" for c in clusters):
        if has_cluster:
            print(f"  WARNING: {os.path.basename(path)} has blank patient_id values "
                  f"— falling back to unclustered DeLong (p-values will be "
                  f"anti-conservative)")
        return labels, scores, None
    return labels, scores, np.array(clusters)


def discover(results_dir, experiments):
    """Find every *_predictions.csv under each requested experiment directory."""
    found = []
    for exp in experiments:
        exp_dir = os.path.join(results_dir, exp)
        if not os.path.isdir(exp_dir):
            print(f"WARNING: no such directory, skipping: {exp_dir}")
            continue
        paths = sorted(glob.glob(os.path.join(exp_dir, "**", "*_predictions.csv"),
                                 recursive=True))
        if not paths:
            print(f"WARNING: no *_predictions.csv under {exp_dir}")
        for p in paths:
            name = os.path.basename(p).replace("_predictions.csv", "")
            found.append({"experiment": exp, "model": name, "path": p})
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def stars(p, q, alpha=0.05):
    if p != p:
        return "  n/a"
    if p >= alpha:
        return "   ns"
    # raw p is significant — flag it if FDR correction takes that away
    if q == q and q >= alpha:
        return "  ns*"
    if p < 0.001:
        return "  ***"
    if p < 0.01:
        return "   **"
    return "    *"


def print_results(rows, alpha, show_naive):
    by_exp = {}
    for r in rows:
        by_exp.setdefault(r["experiment"], []).append(r)

    for exp in sorted(by_exp):
        block = sorted(by_exp[exp], key=lambda r: (r["p"] if r["p"] == r["p"] else 2))
        print(f"\n{'=' * 118}")
        print(f"{exp}  —  DeLong test of AUC vs chance (H0: AUC = 0.5), "
              f"patient-clustered SE, entire test set")
        print(f"{'=' * 118}")
        hdr = (f"{'Model':<52} | {'AUC':>6} | {'95% CI (logit)':>18} | "
               f"{'SE':>7} | {'Z':>7} | {'p':>9} | {'q(BH)':>8} |")
        if show_naive:
            hdr += f" {'SE_unclust':>10} | {'DEff':>5} |"
        print(hdr)
        print("-" * len(hdr))
        for r in block:
            line = (f"{r['model'][:52]:<52} | {r['auc']:>6.4f} | "
                    f"[{r['ci_lower_logit']:.4f}, {r['ci_upper_logit']:.4f}] | "
                    f"{r['se']:>7.4f} | {r['z']:>7.3f} | {r['p']:>9.3g} | "
                    f"{r['q_within_exp']:>8.3g} |")
            if show_naive:
                line += f" {r['se_naive']:>10.4f} | {r['design_effect']:>5.2f} |"
            line += stars(r["p"], r["q_within_exp"], alpha)
            print(line)

        n_sig = sum(1 for r in block if r["q_within_exp"] == r["q_within_exp"]
                    and r["q_within_exp"] < alpha)
        print(f"\n  {len(block)} models tested; {n_sig} significantly different from "
              f"chance at BH-FDR q < {alpha} (within {exp}).")

        # An experiment can span several test sets (e.g. exp5 mixes MIMIC and
        # CheXpert), so summarise each distinct one rather than assuming one.
        shapes = {}
        for r in block:
            key = (r["n_images"], r["n_patients"], r["n_pos"], r["n_neg"])
            shapes.setdefault(key, []).append(r)
        for (n_img, n_pat, n_pos, n_neg), grp in sorted(shapes.items(),
                                                        key=lambda kv: -kv[0][0]):
            pat = f"{n_pat:.0f}" if n_pat == n_pat else "n/a"
            print(f"  Test set ({len(grp)} model(s)): {n_img} images from {pat} "
                  f"patients ({n_pos} positive / {n_neg} negative). "
                  f"Median design effect "
                  f"{np.nanmedian([r['design_effect'] for r in grp]):.2f} "
                  f"→ effective n ~ "
                  f"{np.nanmedian([r['n_effective'] for r in grp]):.0f} images.")

    print(f"\n{'=' * 118}")
    print("'ns*' = raw p < 0.05 but does not survive FDR correction.")
    print("SE is cluster-robust (patient = independent unit). "
          "DEff = Var_clustered / Var_unclustered; DEff > 1 means repeated CXRs "
          "per patient genuinely cost precision.")
    print(f"{'=' * 118}")


FIELDS = ["experiment", "model", "n_images", "n_patients", "n_pos", "n_neg",
          "unit", "auc", "se", "se_naive", "design_effect", "n_effective",
          "z", "p", "q_within_exp", "q_all_tests", "bonferroni_p_all",
          "ci_lower", "ci_upper", "ci_lower_logit", "ci_upper_logit",
          "significant_raw", "significant_fdr", "path"]


def save_csv(rows, output_path):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nResults saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

def self_test():
    """Check the clustered estimator against classical DeLong and against a
    cluster bootstrap, and confirm it is sensitive to real clustering."""
    rng = np.random.RandomState(0)
    print("=" * 70)
    print("Self-test 1: singleton clusters must reproduce classical DeLong")
    print("=" * 70)
    y = rng.binomial(1, 0.45, 800)
    s = rng.normal(y * 0.6, 1.0)
    auc, v10, v01, pm = delong_components(y, s)
    vn = delong_var_naive(auc, v10, v01)
    vc, k = delong_var_clustered(auc, v10, v01, pm, np.arange(len(y)))
    print(f"  AUC                 : {auc:.6f}")
    print(f"  Var classical DeLong: {vn:.8e}")
    print(f"  Var cluster-robust  : {vc:.8e}   ({k} singleton clusters)")
    print(f"  ratio               : {vc / vn:.6f}   (expect ~1.000)")
    assert abs(vc / vn - 1) < 0.01, "clustered estimator must reduce to DeLong"

    print("\n" + "=" * 70)
    print("Self-test 2: real clustering must inflate the variance,")
    print("             and match a cluster bootstrap")
    print("=" * 70)
    # 300 patients, 1-8 correlated images each, label fixed per patient
    pats, ys, ss = [], [], []
    for pid in range(300):
        label = rng.binomial(1, 0.45)
        effect = rng.normal(label * 0.6, 0.9)      # patient-level score level
        for _ in range(rng.randint(1, 9)):
            pats.append(pid)
            ys.append(label)
            ss.append(effect + rng.normal(0, 0.25))  # small within-patient noise
    pats, ys, ss = np.array(pats), np.array(ys), np.array(ss)

    auc, v10, v01, pm = delong_components(ys, ss)
    vn = delong_var_naive(auc, v10, v01)
    vc, k = delong_var_clustered(auc, v10, v01, pm, pats)

    # Independent reference: cluster bootstrap over patients
    uniq = np.unique(pats)
    idx_by_pat = {p: np.where(pats == p)[0] for p in uniq}
    boot = []
    for _ in range(2000):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pat[p] for p in drawn])
        try:
            a, _, _, _ = delong_components(ys[idx], ss[idx])
            boot.append(a)
        except ValueError:
            pass
    vb = np.var(boot, ddof=1)

    print(f"  {len(ys)} images from {k} patients, AUC = {auc:.6f}")
    print(f"  SE classical DeLong : {math.sqrt(vn):.6f}   <- too small, ignores clustering")
    print(f"  SE cluster-robust   : {math.sqrt(vc):.6f}")
    print(f"  SE cluster bootstrap: {math.sqrt(vb):.6f}   <- independent reference")
    print(f"  design effect       : {vc / vn:.3f}")
    print(f"  clustered / bootstrap SE ratio: {math.sqrt(vc / vb):.4f}  (expect ~1.0)")
    assert vc > vn, "clustering must inflate the variance"
    assert 0.85 < math.sqrt(vc / vb) < 1.15, "must agree with the cluster bootstrap"

    print("\nAll self-tests passed.")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeLong test of AUC against the 0.5 chance baseline on the "
                    "entire test set, with a patient-level cluster adjustment."
    )
    parser.add_argument("--results_dir", type=str, default="bootstrap_results",
                        help="Directory containing the per-experiment subdirectories")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["exp1", "exp2", "exp3", "exp4", "exp5"],
                        help="Experiment subdirectories to test")
    parser.add_argument("--predictions", type=str, nargs="+", default=None,
                        help="Test these *_predictions.csv files directly "
                             "instead of scanning --results_dir")
    parser.add_argument("--baseline", type=float, default=0.5,
                        help="Null-hypothesis AUC (default 0.5 = chance)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--label_col", type=str, default="true_label")
    parser.add_argument("--score_col", type=str, default="prob_1",
                        help="Column holding the positive-class score")
    parser.add_argument("--cluster_col", type=str, default="patient_id")
    parser.add_argument("--ignore_clusters", action="store_true",
                        help="Use the classical image-level DeLong variance. "
                             "Anti-conservative here; for comparison only.")
    parser.add_argument("--show_naive", action="store_true",
                        help="Also print the unclustered SE and the design effect")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save the full results CSV")
    parser.add_argument("--self_test", action="store_true",
                        help="Validate the variance estimator and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.predictions:
        targets = [{"experiment": os.path.basename(os.path.dirname(p)) or "single",
                    "model": os.path.basename(p).replace("_predictions.csv", ""),
                    "path": p}
                   for p in args.predictions]
    else:
        targets = discover(args.results_dir, args.experiments)

    if not targets:
        raise SystemExit("No *_predictions.csv files found — nothing to test.")

    print(f"Testing {len(targets)} model(s) against AUC = {args.baseline} "
          f"({'image-level' if args.ignore_clusters else 'patient-clustered'} DeLong SE)\n")

    rows = []
    for t in targets:
        try:
            labels, scores, clusters = load_predictions(
                t["path"], args.label_col, args.score_col, args.cluster_col)
            if args.ignore_clusters:
                clusters = None
            res = test_auc_vs_chance(labels, scores, clusters,
                                     baseline=args.baseline, alpha=args.alpha)
        except (ValueError, OSError) as e:
            print(f"  SKIP {t['model']}: {e}")
            continue
        res.update(t)
        rows.append(res)

    if not rows:
        raise SystemExit("No files could be evaluated.")

    # FDR within each experiment and across every test in the run
    q_all = benjamini_hochberg([r["p"] for r in rows])
    n_all = len(rows)
    for r, q in zip(rows, q_all):
        r["q_all_tests"] = q
        r["bonferroni_p_all"] = min(r["p"] * n_all, 1.0) if r["p"] == r["p"] else float("nan")

    by_exp = {}
    for i, r in enumerate(rows):
        by_exp.setdefault(r["experiment"], []).append(i)
    for exp, idxs in by_exp.items():
        q_exp = benjamini_hochberg([rows[i]["p"] for i in idxs])
        for i, q in zip(idxs, q_exp):
            rows[i]["q_within_exp"] = q

    for r in rows:
        r["significant_raw"] = bool(r["p"] == r["p"] and r["p"] < args.alpha)
        r["significant_fdr"] = bool(r["q_within_exp"] == r["q_within_exp"]
                                    and r["q_within_exp"] < args.alpha)

    print_results(rows, args.alpha, args.show_naive)

    n_sig = sum(r["significant_fdr"] for r in rows)
    print(f"\nOverall: {n_sig}/{len(rows)} models differ from AUC = {args.baseline} "
          f"at BH-FDR q < {args.alpha} (corrected within experiment).")

    if args.output:
        save_csv(rows, args.output)


if __name__ == "__main__":
    main()
