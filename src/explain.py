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

from config import (
    FEATURE_LABELS_PT,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_PATH,
    REASON_GROUPS,
    REASON_TEXT_DIVIDA_SEM_RENDA,
    REASON_TEXTS,
    SEED,
)
from data import clean_static, deduplicate, load_raw, split

SHAP_SAMPLE = 2000
_EXPLAINERS: dict = {}


def shap_values_for(bundle: dict, X: pd.DataFrame):
    """Retorna (SHAP em log-odds, X transformado). Ponto único de cálculo, reusado pelo predict.py."""
    pipe = bundle["raw_pipeline"]
    Xt = pipe[:-1].transform(X)
    if id(pipe[-1]) not in _EXPLAINERS:  # construir o explainer é o passo lento
        _EXPLAINERS[id(pipe[-1])] = shap.TreeExplainer(pipe[-1])
    explainer = _EXPLAINERS[id(pipe[-1])]
    with warnings.catch_warnings():  # aviso informativo do shap sobre o formato de saída do LightGBM
        warnings.simplefilter("ignore", UserWarning)
        sv = explainer.shap_values(Xt)
    if isinstance(sv, list):  # versões antigas devolvem [classe0, classe1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xt


def explanation_for(bundle: dict, X: pd.DataFrame) -> shap.Explanation:
    """Explanation da 1ª linha de X (nomes em PT), pronta para o waterfall. Unidade: log-odds do modelo cru."""
    sv, Xt = shap_values_for(bundle, X.iloc[:1])
    base = float(np.ravel(_EXPLAINERS[id(bundle["raw_pipeline"][-1])].expected_value)[-1])
    return shap.Explanation(
        values=sv[0],
        base_values=base,
        data=Xt.iloc[0].to_numpy(),
        feature_names=[FEATURE_LABELS_PT.get(c, c) for c in Xt.columns],
    )


def waterfall_figure(expl: shap.Explanation, title: str = ""):
    plt.figure()
    shap.plots.waterfall(expl, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    if title:
        plt.title(title)
    fig.tight_layout()
    return fig


FLAG_FEATURES = {"renda_ausente", "atraso_codigo_especial", "teve_atraso_grave", "sem_uso_rotativo"}
# Guarda de segurança: desde a v0.2 a idade não está no modelo (ADR-18), então esta regra fica inerte; ela só
# impede que a idade vire motivo se algum dia voltar a ser variável do modelo.
NOT_DISPLAYED_AS_REASON = {"idade"}


def _group_text(group: str, idx: dict, values) -> str | None:
    """Texto do motivo de um grupo. Sem `values`, usa o texto genérico. Devolve None quando o grupo não deve
    virar motivo (ex.: sem nenhum atraso; renda não informada, que já tem motivo próprio)."""
    if group == "atrasos":
        if values is None:
            return REASON_TEXTS["total_atrasos"]
        v = lambda n: values[idx[n]] if n in idx else 0
        if v("atrasos_90") > 0 or v("teve_atraso_grave") == 1:
            return REASON_TEXTS["atrasos_90"]
        if v("atrasos_60_89") > 0:
            return REASON_TEXTS["atrasos_60_89"]
        if v("atrasos_30_59") > 0:
            return REASON_TEXTS["atrasos_30_59"]
        return None
    if group == "renda":
        if values is not None and "renda_ausente" in idx and values[idx["renda_ausente"]] == 1:
            return None
        return REASON_TEXTS["renda_mensal"]
    return None


def reason_codes_with_impact(shap_row, feature_names, top: int = 3, values=None) -> list[tuple[str, float]]:
    """Até `top` motivos DISTINTOS, só de fatores que AUMENTAM o risco, em ordem decrescente (ADR-21).
    Variáveis de um mesmo grupo (REASON_GROUPS) somam o SHAP e viram um único motivo. Com `values`
    (linha transformada), flags só entram se estiverem ATIVAS, o texto de atrasos segue o atraso mais grave
    presente e, com renda não informada, a dívida é descrita como valor (não como razão).
    Devolve (texto, SHAP em log-odds do fator ou grupo)."""
    shap_row = np.asarray(shap_row, dtype=float)
    idx = {n: i for i, n in enumerate(feature_names)}
    vals = None if values is None else np.asarray(values, dtype=float)
    items, grouped = [], set()
    for g, members in REASON_GROUPS.items():
        present = [m for m in members if m in idx]
        if present:
            items.append((float(shap_row[[idx[m] for m in present]].sum()), "grupo", g))
            grouped.update(present)
    items += [(float(shap_row[i]), "var", n) for n, i in idx.items() if n not in grouped]
    items.sort(key=lambda t: -t[0])

    reasons: list[tuple[str, float]] = []
    for impact, kind, key in items:
        if impact <= 0 or len(reasons) == top:
            break
        if kind == "grupo":
            text = _group_text(key, idx, vals)
        else:
            if key in NOT_DISPLAYED_AS_REASON:
                continue
            if vals is not None and key in FLAG_FEATURES and vals[idx[key]] != 1:
                continue
            if key == "utilizacao_limite" and vals is not None and "sem_uso_rotativo" in idx and vals[idx["sem_uso_rotativo"]] == 1:
                continue  # utilização zero nunca é descrita como "uso elevado do limite"
            text = REASON_TEXTS.get(key)
            if key == "razao_divida" and vals is not None and "renda_ausente" in idx and vals[idx["renda_ausente"]] == 1:
                text = REASON_TEXT_DIVIDA_SEM_RENDA
        if text and text not in [r[0] for r in reasons]:
            reasons.append((text, impact))
    return reasons


def reason_codes(shap_row, feature_names, top: int = 3, values=None) -> list[str]:
    return [t for t, _ in reason_codes_with_impact(shap_row, feature_names, top, values)]


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
    plt.gca().set_xlabel("Impacto no risco (SHAP, log-odds)")
    cbar = plt.gcf().axes[-1]  # rótulos em português (o shap só gera em inglês)
    cbar.set_ylabel("Valor da variável")
    cbar.set_yticklabels(["Baixo", "Alto"])
    plt.title("SHAP: impacto de cada variável no risco (log-odds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(sv, Xt_pt, plot_type="bar", show=False, max_display=16)
    plt.gca().set_xlabel("Importância média |SHAP| (log-odds)")
    plt.title("SHAP: importância média (|SHAP|)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=150)
    plt.close()

    imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)

    # --- local: motivos nas decisões do teste, pela decisão em 3 faixas do produto (ADR-20, ADR-21) ---
    from evaluate import band_or_cut  # import tardio, como o de predict abaixo

    pd_te = bundle["calibrator"].predict(bundle["raw_pipeline"].predict_proba(X_te)[:, 1])
    t = bundle["threshold"]
    lo, hi = band_or_cut(bundle.get("review_band"), t)
    names = list(bundle["feature_names_out"])
    stats, freq = {}, {}
    examples = []
    for label, mask in (("recusar", pd_te > hi), ("revisar", (pd_te >= lo) & (pd_te <= hi))):
        sv_r, Xt_r = shap_values_for(bundle, X_te[mask])
        vals_r = Xt_r.to_numpy()
        rc = [reason_codes(r, names, values=v) for r, v in zip(sv_r, vals_r)]
        n_reasons = np.array([len(r) for r in rc])
        stats[label] = {
            "n": int(mask.sum()),
            "com_3_motivos": int((n_reasons == 3).sum()),
            "com_2_motivos": int((n_reasons == 2).sum()),
            "com_1_motivo": int((n_reasons == 1).sum()),
            "com_0_motivos": int((n_reasons == 0).sum()),
        }
        cont = pd.Series([m for r in rc for m in r]).value_counts()
        prim = pd.Series([r[0] for r in rc if r]).value_counts()
        freq[label] = {m: {"qualquer_posicao": int(cont[m]), "primeiro_motivo": int(prim.get(m, 0))} for m in cont.index}
        if label == "recusar" and mask.any():
            for k in np.argsort(-pd_te[mask])[[0, len(sv_r) // 2, -1]]:
                examples.append({"pd": float(pd_te[mask][k]), "motivos": rc[k]})

    # T5.5: waterfall dos 3 perfis da demo (import tardio: predict importa este módulo)
    from config import PROFILES
    from predict import _model_frame, _validate_frame

    for nome, perfil in PROFILES.items():
        Xp = _model_frame(_validate_frame(pd.DataFrame([perfil])), bundle)
        fig = waterfall_figure(explanation_for(bundle, Xp), f"Perfil {nome}")
        fig.savefig(FIGURES_DIR / f"shap_waterfall_{nome.lower().replace('í', 'i')}.png", dpi=150)
        plt.close(fig)

    out = {
        "shap_amostra": SHAP_SAMPLE,
        "importancia_media_abs_shap": {k: float(v) for k, v in imp.items()},
        "reason_codes_teste": stats,
        "frequencia_motivos_teste": freq,
        "exemplos": examples,
        "corte": t,
        "review_band": [lo, hi],
    }
    (METRICS_DIR / "explain.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Top-8 |SHAP| médio:")
    print(imp.head(8).round(4).to_string())
    print(json.dumps(stats, indent=2))
    print(json.dumps(freq, indent=2, ensure_ascii=False))
    for e in examples:
        print(f"PD {e['pd']:.3f}: {e['motivos']}")


if __name__ == "__main__":
    main()
