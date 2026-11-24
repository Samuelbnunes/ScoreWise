"""Treino: CV 5-fold -> OOF cruas -> calibrador (CALIBRATION_METHOD) -> corte por custo -> fit final -> model.joblib.
v0.2.0: sem class_weight, sem idade e sem divida_mensal, com restrições monotônicas no LightGBM.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from calibration import fit_calibrator
from config import (
    CALIBRATION_METHOD,
    COST_MATRIX,
    CV_FOLDS,
    LGBM_PARAMS,
    METRICS_DIR,
    MODEL_EXCLUDED_FEATURES,
    MODEL_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    MONOTONE,
    MONOTONE_METHOD,
    RAW_FILE,
    REVIEW_PARAMS,
    SEED,
)
from data import clean_static, deduplicate, file_sha256, load_raw, split
from evaluate import (
    CANDIDATES_PATH,
    compute_metrics,
    cost_threshold,
    f1_threshold,
    profit_at,
    review_band,
    theoretical_threshold,
)
from features import build_preprocessor, output_feature_names

METRIC_KEYS = ["auc", "gini", "ks", "pr_auc", "brier"]


def monotone_vector(drop_cols: tuple = MODEL_EXCLUDED_FEATURES) -> list[int]:
    """MONOTONE alinhado à ordem das colunas que o pré-processador entrega ao LightGBM."""
    return [MONOTONE.get(c, 0) for c in output_feature_names(drop_cols)]


def build_lgbm(drop_cols: tuple = MODEL_EXCLUDED_FEATURES, **override) -> LGBMClassifier:
    kw = dict(
        **LGBM_PARAMS,
        monotone_constraints=monotone_vector(drop_cols),
        monotone_constraints_method=MONOTONE_METHOD,
        random_state=SEED,
        n_jobs=-1,
        deterministic=True,  # resultados idênticos entre execuções
        force_row_wise=True,
        verbose=-1,
    )
    kw.update(override)
    return LGBMClassifier(**kw)


def build_candidates(drop_cols: tuple = MODEL_EXCLUDED_FEATURES) -> dict:
    """Mesmas variáveis para os três candidatos. Sem class_weight (ADR-15)."""
    return {
        "dummy": Pipeline([("pre", build_preprocessor("tree", drop_cols)), ("clf", DummyClassifier(strategy="prior"))]),
        "logreg": Pipeline(
            [
                ("pre", build_preprocessor("linear", drop_cols)),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        ),
        "lgbm": Pipeline([("pre", build_preprocessor("tree", drop_cols)), ("clf", build_lgbm(drop_cols))]),
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
        calibrator = fit_calibrator(CALIBRATION_METHOD, oof[name], y_tr)
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
        assert list(final[:-1].get_feature_names_out()) == output_feature_names(MODEL_EXCLUDED_FEATURES), \
            "ordem das colunas diverge de output_feature_names (restrições monotônicas desalinhadas)"
        cand_out[name] = {"pipeline": final, "calibrator": calibrator, "thresholds": thresholds}

    joblib.dump(cand_out, CANDIDATES_PATH)

    b = cand_out[best]
    bundle = {
        "best_name": best,
        "calibrator": b["calibrator"],
        "raw_pipeline": b["pipeline"],
        "threshold": b["thresholds"]["cost"],
        "thresholds": b["thresholds"],
        "review_band": review_band(COST_MATRIX, REVIEW_PARAMS),  # (lo, hi) ou None (ADR-20)
        "review_params": dict(REVIEW_PARAMS),
        "feature_names_in": list(X_tr.columns),
        "feature_names_out": list(b["pipeline"][:-1].get_feature_names_out()),
        "metrics_cv": cv[best],
        "calibration_method": CALIBRATION_METHOD,
        "excluded_features": list(MODEL_EXCLUDED_FEATURES),
        "monotone_constraints": {c: v for c, v in zip(output_feature_names(MODEL_EXCLUDED_FEATURES), monotone_vector()) if v},
        "monotone_method": MONOTONE_METHOD,
        "lgbm_params": dict(LGBM_PARAMS),
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
    rb = bundle["review_band"]
    print(f"faixa REVISAR por valor esperado: {'vazia (decisão em 2 faixas)' if rb is None else f'[{rb[0]:.3f}; {rb[1]:.3f}]'}"
          f" | parâmetros ilustrativos: {REVIEW_PARAMS}")
    print("feature_names_out:", bundle["feature_names_out"])
    print(f"salvo: {MODEL_PATH} | tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
