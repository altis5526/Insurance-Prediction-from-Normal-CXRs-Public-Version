"""
DeLong test of AUC against the chance baseline (AUC = 0.5) for the classical ML
models in ml_analysis_predefined_val.ipynb, with the same patient-level cluster
adjustment used for the deep-learning experiments.

The notebook writes bootstrap CSVs to bootstrap_results_ml/ but never saves
per-sample predictions, which is what an analytic DeLong test needs. This script
rebuilds the notebook's pipeline exactly, refits each model on X_train with the
hyperparameters the notebook's GridSearchCV selected, then:

  1. writes <Model>_predictions.csv into bootstrap_results_ml/ using the SAME
     schema bootstrap_evaluate.py produces, so statistical_testing_DeLong.py
     can also be pointed at them directly, and
  2. runs the patient-clustered DeLong test against AUC = 0.5.

All variance machinery is imported from statistical_testing_DeLong.py — there is
exactly one implementation of the statistics, shared with the CNN experiments.
See that file's docstring for the derivation. In brief: the AUC is a two-sample
U-statistic whose influence function is built from the DeLong placement values
V10/V01; contributions are summed within patient and the variance is taken
across patients, so the independent unit is the patient rather than the image.

TIES MATTER MUCH MORE HERE THAN FOR THE CNNs
--------------------------------------------
These models see only three categorical features (gender, age_mapped,
race_mapped), so there are at most 3 x 3 x 2 = 18 distinct feature rows and
therefore at most 18 distinct predicted probabilities across ~1900 test images.
The CNN prediction files were effectively tie-free (1964 distinct scores out of
1965); here whole blocks of hundreds of images share an identical score.

Tied scores are genuine 1/2 credit in the Mann-Whitney sense, and both the AUC
and the placement values are computed with the midrank algorithm, which handles
them exactly. Two consequences worth knowing when reading the output:

  - A tie block gives every image in it the same placement value, which REDUCES
    the spread of V10/V01 and so shrinks the DeLong variance. This is correct,
    not optimistic: a model that cannot separate within a demographic cell
    genuinely has less to be uncertain about.
  - Because each patient's images are drawn from a single demographic cell,
    every image of a patient receives an IDENTICAL placement value. The
    within-patient correlation is therefore exactly 1, which is the strongest
    possible clustering and makes the patient adjustment matter more here than
    for the CNNs, where it was partial.

Usage:
    # Refit, export predictions, and test
    python statistical_testing_DeLong_ml.py

    # Also print the unclustered SE and the design effect
    python statistical_testing_DeLong_ml.py --show_naive

    # Verify the refits reproduce the notebook's reported test AUCs
    python statistical_testing_DeLong_ml.py --verify
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from statistical_testing_DeLong import (
    benjamini_hochberg,
    delong_components,
    test_auc_vs_chance,
)

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing — kept identical to ml_analysis_predefined_val.ipynb cell 1
# ──────────────────────────────────────────────────────────────────────────────

PATIENT_ID_CANDIDATES = ["subject_id_x", "subject_id", "subject_idx"]


def map_insurance_type(insurance):
    if insurance in ["Medicaid", "Medicare"]:
        return "public"
    elif insurance == "Private":
        return "private"
    return "others"


def map_age(age):
    if age < 40:
        return "young"
    elif 40 <= age < 50:
        return "adult"
    elif 50 <= age < 64:
        return "senior"
    return "elderly"


def map_race(race):
    if race == "WHITE":
        return "white"
    elif race == "BLACK":
        return "black"
    return "others"


def find_patient_col(df):
    for name in PATIENT_ID_CANDIDATES:
        if name in df.columns:
            return name
    return None


def preprocess_raw(df):
    df = df.copy()
    df["insurance_mapped"] = df["new_insurance_type"].apply(map_insurance_type)
    df["age_mapped"] = df["anchor_age"].apply(map_age)
    df["race_mapped"] = df["race"].apply(map_race)
    pid_col = find_patient_col(df)
    if pid_col is not None:
        df["patient_id"] = df[pid_col].astype(str)
    # Carry dicom_id through so the exported predictions can be joined per-image
    # against the CNN prediction files, which use it as their sample_id.
    if "dicom_id" in df.columns:
        df["dicom_id"] = df["dicom_id"].astype(str)
    df = df[df["insurance_mapped"].isin(["public", "private"])]
    df = df[df["age_mapped"].isin(["young", "adult", "senior"])]
    keep = ["age_mapped", "race_mapped", "gender", "insurance_mapped",
            "patient_id", "dicom_id"]
    return df[[c for c in df.columns if c in keep]]


def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    df["insurance_mapped"] = le.fit_transform(df["insurance_mapped"])
    df["age_mapped"] = le.fit_transform(df["age_mapped"])
    df["race_mapped"] = le.fit_transform(df["race_mapped"])
    return df[["gender", "age_mapped", "race_mapped"]], df["insurance_mapped"]


def prepare_split(df):
    proc = preprocess_raw(df).reset_index(drop=True)
    X, y = encode_features(proc)
    patients = proc["patient_id"] if "patient_id" in proc.columns else None
    dicom = proc["dicom_id"] if "dicom_id" in proc.columns else None
    return (X.reset_index(drop=True), y.reset_index(drop=True),
            patients.reset_index(drop=True) if patients is not None else None,
            dicom.reset_index(drop=True) if dicom is not None else None)


# ──────────────────────────────────────────────────────────────────────────────
# Models — hyperparameters as selected by the notebook's PredefinedSplit search
# ──────────────────────────────────────────────────────────────────────────────

# Reported test AUC in the notebook's stored output, used by --verify
NOTEBOOK_TEST_AUC = {
    "Random Forest": 0.6141,
    "XGBoost":       0.6125,
    "CatBoost":      0.6148,
    "KNN":           0.6108,
    "Decision Tree": 0.6150,
}


def build_models(y_train):
    """Recreate the notebook's five fitted models with its selected hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb
    from catboost import CatBoostClassifier

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    return {
        "Random Forest": RandomForestClassifier(
            bootstrap=True, class_weight=None, max_depth=10,
            min_samples_leaf=1, min_samples_split=2, n_estimators=300,
            random_state=42),
        "XGBoost": xgb.XGBClassifier(
            colsample_bytree=0.8, learning_rate=0.1, max_depth=3,
            n_estimators=100, scale_pos_weight=scale_pos_weight, subsample=0.8,
            random_state=42, eval_metric="logloss", verbosity=0),
        "CatBoost": CatBoostClassifier(
            auto_class_weights="Balanced", bagging_temperature=0.5,
            border_count=64, depth=6, iterations=500, l2_leaf_reg=5,
            learning_rate=0.05, random_state=42, verbose=0, eval_metric="AUC"),
        "KNN": KNeighborsClassifier(
            metric="euclidean", n_neighbors=9, weights="uniform"),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=None, criterion="gini", max_depth=10,
            min_samples_leaf=1, min_samples_split=2, random_state=42),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prediction export — same schema as bootstrap_evaluate.py save_predictions()
# ──────────────────────────────────────────────────────────────────────────────

GENDER_NAMES = {0: "Female", 1: "Male"}
AGE_NAMES = {0: "Adult", 1: "Senior", 2: "Young"}
RACE_NAMES = {0: "Black", 1: "Others", 2: "White"}


def save_predictions(X_test, y_test, patients, dicom, y_prob, y_pred,
                     output_dir, model_name):
    """Write <Model>_predictions.csv matching bootstrap_evaluate.py's columns."""
    os.makedirs(output_dir, exist_ok=True)
    safe = model_name.replace(" ", "_")
    path = os.path.join(output_dir, f"{safe}_predictions.csv")

    fields = (["sample_id", "patient_id", "true_label", "pred_label", "correct"]
              + ["logit_0", "logit_1", "prob_0", "prob_1"]
              + ["gender", "age", "race", "gender_name", "age_name", "race_name"])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(y_test)):
            p1 = float(y_prob[i])
            p0 = 1.0 - p1
            g, a, r = (int(X_test["gender"].iloc[i]),
                       int(X_test["age_mapped"].iloc[i]),
                       int(X_test["race_mapped"].iloc[i]))
            writer.writerow({
                "sample_id": dicom.iloc[i] if dicom is not None else i,
                "patient_id": patients.iloc[i] if patients is not None else "",
                "true_label": int(y_test.iloc[i]),
                "pred_label": int(y_pred[i]),
                "correct": int(y_pred[i] == y_test.iloc[i]),
                # These models expose no logits; log-probabilities keep the
                # column shape without implying a scale the model does not have.
                "logit_0": float(np.log(max(p0, 1e-12))),
                "logit_1": float(np.log(max(p1, 1e-12))),
                "prob_0": p0,
                "prob_1": p1,
                "gender": g, "age": a, "race": r,
                "gender_name": GENDER_NAMES.get(g, "Unknown"),
                "age_name": AGE_NAMES.get(a, "Unknown"),
                "race_name": RACE_NAMES.get(r, "Unknown"),
            })
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def describe_ties(scores):
    """Summarise the tie structure, which drives how DeLong behaves here."""
    uniq, counts = np.unique(scores, return_counts=True)
    return len(uniq), int(counts.max())


def print_results(rows, alpha, show_naive):
    print(f"\n{'=' * 112}")
    print("bootstrap_results_ml  —  DeLong test of AUC vs chance (H0: AUC = 0.5)")
    print("Patient-clustered SE, entire test set, no resampling")
    print(f"{'=' * 112}")
    hdr = (f"{'Model':<16} | {'AUC':>6} | {'95% CI (logit)':>18} | {'SE':>7} | "
           f"{'Z':>7} | {'p':>10} | {'q(BH)':>10} |")
    if show_naive:
        hdr += f" {'SE_unclust':>10} | {'DEff':>5} |"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["p"]):
        line = (f"{r['model']:<16} | {r['auc']:>6.4f} | "
                f"[{r['ci_lower_logit']:.4f}, {r['ci_upper_logit']:.4f}] | "
                f"{r['se']:>7.4f} | {r['z']:>7.3f} | {r['p']:>10.3g} | "
                f"{r['q']:>10.3g} |")
        if show_naive:
            line += f" {r['se_naive']:>10.4f} | {r['design_effect']:>5.2f} |"
        line += "  ***" if r["q"] < 0.001 else ("   **" if r["q"] < 0.01 else
                ("    *" if r["q"] < alpha else "   ns"))
        print(line)

    ex = rows[0]
    print(f"\n  Test set: {ex['n_images']} images from {ex['n_patients']:.0f} patients "
          f"({ex['n_pos']} public / {ex['n_neg']} private).")
    print(f"  Distinct predicted probabilities per model: "
          f"{', '.join(str(r['n_distinct_scores']) for r in rows)} "
          f"(largest tie block: {max(r['max_tie_block'] for r in rows)} images).")
    print(f"  Median design effect {np.median([r['design_effect'] for r in rows]):.2f} "
          f"→ effective n ~ {np.median([r['n_effective'] for r in rows]):.0f} images.")
    n_sig = sum(r["q"] < alpha for r in rows)
    print(f"  {n_sig}/{len(rows)} models differ from chance at BH-FDR q < {alpha}.")
    print(f"{'=' * 112}")


FIELDS = ["model", "n_images", "n_patients", "n_pos", "n_neg",
          "n_distinct_scores", "max_tie_block", "auc", "se", "se_naive",
          "design_effect", "n_effective", "z", "p", "q", "bonferroni_p",
          "ci_lower", "ci_upper", "ci_lower_logit", "ci_upper_logit",
          "significant_raw", "significant_fdr", "predictions_path"]


def main():
    parser = argparse.ArgumentParser(
        description="Patient-clustered DeLong test vs AUC = 0.5 for the "
                    "ml_analysis_predefined_val.ipynb models")
    parser.add_argument("--train_csv", type=str,
                        default="./insurance_dataset_8_1_1_PMMthree_train_medgemmaChecked.csv")
    parser.add_argument("--test_csv", type=str,
                        default="./insurance_dataset_8_1_1_PMMthree_test_medgemmaChecked.csv")
    parser.add_argument("--output_dir", type=str, default="bootstrap_results_ml")
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ignore_clusters", action="store_true",
                        help="Use the image-level DeLong variance (anti-conservative)")
    parser.add_argument("--show_naive", action="store_true",
                        help="Also print the unclustered SE and the design effect")
    parser.add_argument("--verify", action="store_true",
                        help="Check the refits reproduce the notebook's test AUCs")
    parser.add_argument("--output", type=str,
                        default="bootstrap_results_ml/delong_vs_chance_ml.csv")
    args = parser.parse_args()

    X_train, y_train, _, _ = prepare_split(pd.read_csv(args.train_csv))
    X_test, y_test, patients_test, dicom_test = prepare_split(
        pd.read_csv(args.test_csv))

    if patients_test is None:
        print("WARNING: no patient id column in the test CSV — falling back to "
              "image-level DeLong (p-values will be anti-conservative)")

    print(f"Train: {len(X_train)} images    Test: {len(X_test)} images"
          + (f" / {patients_test.nunique()} patients" if patients_test is not None else ""))
    print(f"Testing H0: AUC = {args.baseline}  "
          f"({'image-level' if args.ignore_clusters else 'patient-clustered'} DeLong SE)\n")

    rows = []
    for name, model in build_models(y_train).items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        path = save_predictions(X_test, y_test, patients_test, dicom_test,
                                y_prob, y_pred, args.output_dir, name)

        clusters = None if (args.ignore_clusters or patients_test is None) \
            else patients_test.to_numpy()
        res = test_auc_vs_chance(y_test.to_numpy(), y_prob, clusters,
                                 baseline=args.baseline, alpha=args.alpha)
        n_uniq, max_tie = describe_ties(y_prob)
        res.update({"model": name, "predictions_path": path,
                    "n_distinct_scores": n_uniq, "max_tie_block": max_tie})
        rows.append(res)

        note = ""
        if args.verify:
            expected = NOTEBOOK_TEST_AUC[name]
            ok = abs(res["auc"] - expected) < 5e-4
            note = (f"   notebook {expected:.4f}  "
                    f"{'MATCH' if ok else 'MISMATCH'}")
        print(f"  {name:<16} AUC {res['auc']:.4f}   "
              f"{n_uniq:>3} distinct scores, largest tie block {max_tie:>4}{note}")

    qs = benjamini_hochberg([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q"] = q
        r["bonferroni_p"] = min(r["p"] * len(rows), 1.0)
        r["significant_raw"] = bool(r["p"] < args.alpha)
        r["significant_fdr"] = bool(q < args.alpha)

    print_results(rows, args.alpha, args.show_naive)

    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in FIELDS})
        print(f"\nResults saved to: {args.output}")
        print(f"Per-sample predictions written to: {args.output_dir}/<Model>_predictions.csv")
        print("  (statistical_testing_DeLong.py --predictions can be pointed at these too)")


if __name__ == "__main__":
    main()
