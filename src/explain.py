"""SHAP global/local e reason codes. O SHAP é calculado sobre o modelo CRU (ADR-05): a calibração
é monotônica e não muda a ordem dos fatores. Valores SHAP estão em log-odds do modelo cru."""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import FEATURE_LABELS_PT, FIGURES_DIR, METRICS_DIR, MODEL_PATH, REASON_TEXTS, SEED
from data import clean_static, deduplicate, load_raw, split

SHAP_SAMPLE = 2000


def shap_values_for(bundle: dict, X: pd.DataFrame):
    """Retorna (SHAP em log-odds, X transformado). Ponto único de cálculo, reusado pelo predict.py."""
    pipe = bundle["raw_pipeline"]
    Xt = pipe[:-1].transform(X)
    with warnings.catch_warnings():  # aviso informativo do shap sobre o formato de saída do LightGBM
        warnings.simplefilter("ignore", UserWarning)
        sv = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
    if isinstance(sv, list):  # versões antigas devolvem [classe0, classe1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xt


def reason_codes(shap_row, feature_names, top: int = 3) -> list[str]:
    """Só contribuições POSITIVAS (que aumentam o risco), em ordem decrescente; textos repetidos
    (ex.: renda_mensal e renda_per_capita) aparecem uma vez."""
    order = np.argsort(-np.asarray(shap_row))
    reasons: list[str] = []
    for i in order:
        if shap_row[i] <= 0 or len(reasons) == top:
            break
        text = REASON_TEXTS.get(feature_names[i])
        if text and text not in reasons:
            reasons.append(text)
    return reasons


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    df = deduplicate(clean_static(load_raw()), verbose=False)
    _, X_te, _, y_te = split(df)

    # --- global (amostra de 2.000 por desempenho) ---
    samp = X_te.sample(SHAP_SAMPLE, random_state=SEED)
    sv, Xt = shap_values_for(bundle, samp)
    Xt_pt = Xt.rename(columns=FEATURE_LABELS_PT)

    plt.figure()
    shap.summary_plot(sv, Xt_pt, show=False, max_display=16)
    plt.title("SHAP: impacto de cada variável no risco (log-odds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(sv, Xt_pt, plot_type="bar", show=False, max_display=16)
    plt.title("SHAP: importância média (|SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150)
    plt.close()

    imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)

    # --- local: todas as recusas do teste (DoD: 3 motivos não vazios) ---
    pd_te = bundle["calibrator"].predict(bundle["raw_pipeline"].predict_proba(X_te)[:, 1])
    t, delta = bundle["threshold"], bundle["review_band"]
    names = list(bundle["feature_names_out"])
    stats = {}
    examples = []
    for label, mask in (("recusar_pd_ge_corte", pd_te >= t), ("recusar_faixa_pd_gt_corte_mais_delta", pd_te > t + delta)):
        sv_r, _ = shap_values_for(bundle, X_te[mask])
        n_reasons = np.array([len(reason_codes(r, names)) for r in sv_r])
        stats[label] = {
            "n_recusas": int(mask.sum()),
            "com_3_motivos": int((n_reasons == 3).sum()),
            "com_menos_de_3": int((n_reasons < 3).sum()),
            "com_0_motivos": int((n_reasons == 0).sum()),
        }
        if label == "recusar_pd_ge_corte":
            for k in np.argsort(-pd_te[mask])[[0, len(sv_r) // 2, -1]]:
                examples.append({"pd": float(pd_te[mask][k]), "motivos": reason_codes(sv_r[k], names)})

    out = {
        "shap_amostra": SHAP_SAMPLE,
        "importancia_media_abs_shap": {k: float(v) for k, v in imp.items()},
        "reason_codes_recusas_teste": stats,
        "exemplos": examples,
        "corte": t,
        "review_band": delta,
    }
    (METRICS_DIR / "explain.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Top-8 |SHAP| médio:")
    print(imp.head(8).round(4).to_string())
    print(json.dumps(stats, indent=2))
    for e in examples:
        print(f"PD {e['pd']:.3f}: {e['motivos']}")


if __name__ == "__main__":
    main()
