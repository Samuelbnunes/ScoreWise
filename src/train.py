"""Treino: CV 5-fold -> OOF cruas -> calibrador isotônico -> corte por custo -> fit final -> model.joblib.
O conjunto de TESTE não é lido aqui."""
import json
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import shap
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import (
    COST_MATRIX,
    CV_FOLDS,
    METRICS_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    RAW_FILE,
    REVIEW_BAND,
    SEED,
)
from data import clean_static, deduplicate, file_sha256, load_raw, split
from evaluate import (
    CANDIDATES_PATH,
    compute_metrics,
    cost_threshold,
    f1_threshold,
    profit_at,
    theoretical_threshold,
)
from features import build_preprocessor

METRIC_KEYS = ["auc", "gini", "ks", "pr_auc", "brier"]


def build_candidates() -> dict:
    return {
        "dummy": Pipeline([("pre", build_preprocessor("tree")), ("clf", DummyClassifier(strategy="prior"))]),
        "logreg": Pipeline(
            [
                ("pre", build_preprocessor("linear")),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
            ]
        ),
        "lgbm": Pipeline(
            [
                ("pre", build_preprocessor("tree")),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=31,
                        min_child_samples=100,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.8,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                        deterministic=True,  # resultados idênticos entre execuções
                        force_row_wise=True,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = deduplicate(clean_static(load_raw()))
    X_tr, _, y_tr, _ = split(df)  # teste descartado: só entra em evaluate.py
    print(f"treino: {X_tr.shape} | taxa de mau: {y_tr.mean():.4%}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    candidates = build_candidates()
    cv, oof = {}, {}

    for name, pipe in candidates.items():
        p_oof = np.zeros(len(y_tr))
        folds = []
        for tr, va in skf.split(X_tr, y_tr):
            m = clone(pipe).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            p = m.predict_proba(X_tr.iloc[va])[:, 1]
            p_oof[va] = p
            folds.append(compute_metrics(y_tr.iloc[va], p))
        oof[name] = p_oof
        cv[name] = {
            k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds], ddof=1)),
                "folds": [float(f[k]) for f in folds]}
            for k in METRIC_KEYS
        }
        print(f"[CV] {name:7s} AUC {cv[name]['auc']['mean']:.4f} ± {cv[name]['auc']['std']:.4f}")

    best = max(cv, key=lambda n: cv[n]["auc"]["mean"])
    theo = theoretical_threshold()
    print(f"melhor por AUC médio na CV: {best}")

    cand_out = {}
    for name, pipe in candidates.items():
        # Calibrador ajustado nas OOF cruas: cada probabilidade vem de um modelo que não viu aquela linha.
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof[name], y_tr)
        pd_oof = calibrator.predict(oof[name])
        thresholds = {
            "cost": cost_threshold(y_tr, pd_oof),
            "f1": f1_threshold(y_tr, pd_oof),
            "default_0.5": 0.5,
            "theoretical": theo,
        }
        cv[name]["brier_calibrated_oof_insample"] = float(compute_metrics(y_tr, pd_oof)["brier"])
        cv[name]["thresholds"] = thresholds
        cv[name]["profit_per_proposal_oof_at_cost"] = profit_at(y_tr, pd_oof, thresholds["cost"]) / len(y_tr)
        final = clone(pipe).fit(X_tr, y_tr)
        cand_out[name] = {"pipeline": final, "calibrator": calibrator, "thresholds": thresholds}

    joblib.dump(cand_out, CANDIDATES_PATH)

    b = cand_out[best]
    bundle = {
        "best_name": best,
        "calibrator": b["calibrator"],
        "raw_pipeline": b["pipeline"],
        "threshold": b["thresholds"]["cost"],
        "thresholds": b["thresholds"],
        "review_band": REVIEW_BAND,
        "feature_names_in": list(X_tr.columns),
        "feature_names_out": list(b["pipeline"][:-1].get_feature_names_out()),
        "metrics_cv": cv[best],
        "cost_matrix": COST_MATRIX,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": file_sha256(RAW_FILE),
        "seed": SEED,
        "lib_versions": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__},
    }
    joblib.dump(bundle, MODEL_PATH)

    results = {
        "best_model": best,
        "n_train": int(len(y_tr)),
        "cv_folds": CV_FOLDS,
        "theoretical_threshold": theo,
        "models": cv,
        "data_sha256": bundle["data_sha256"],
        "seed": SEED,
    }
    (METRICS_DIR / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    t = bundle["thresholds"]
    print(f"corte por custo (OOF calibrado): {t['cost']:.3f} | F1: {t['f1']:.3f} | teórico: {theo:.4f}")
    print("feature_names_out:", bundle["feature_names_out"])
    print(f"salvo: {MODEL_PATH} | tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
