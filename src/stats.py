"""Rigor estatístico: bootstrap no teste (T8.1/T8.2), ablação da idade (T8.3, Q5), sensibilidade da matriz de
custos (T8.4, Q6) e da faixa REVISAR (E3). Saída: reports/metrics/stats.json.
v0.2.0: o modelo final NÃO usa idade (ADR-18); a ablação compara com o contrafactual "com idade" (mesmo protocolo).
O teste é lido só para avaliar; corte e calibrador vêm do treino (OOF), como no train.py."""
import json

import joblib
import numpy as np
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from calibration import fit_calibrator
from config import CALIBRATION_METHOD, COST_MATRIX, CV_FOLDS, METRICS_DIR, MODEL_EXCLUDED_FEATURES, MODEL_PATH, SEED
from data import clean_static, deduplicate, load_raw, split
from evaluate import (
    CANDIDATES_PATH,
    THRESHOLD_GRID,
    cost_threshold,
    portfolio_stats,
    profit_at,
    profit_curve,
    review_sensitivity,
    theoretical_threshold,
)
from fairness import group_table
from features import build_preprocessor
from train import build_candidates, build_lgbm

B = 1000
LOSSES = (-3, -5, -10)


def oof_and_final(pipe, X, y):
    """Mesmo protocolo do train.py: CV 5-fold -> OOF cruas -> calibrador (CALIBRATION_METHOD); fit final no treino todo."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    oof, aucs = np.zeros(len(y)), []
    for tr, va in skf.split(X, y):
        p = clone(pipe).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(float(roc_auc_score(y.iloc[va], p)))
    cal = fit_calibrator(CALIBRATION_METHOD, oof, y)
    return oof, cal, clone(pipe).fit(X, y), aucs


def _ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def _summ(a, est):
    lo, hi = _ci(a)
    return {"est": float(est), "ci95": [lo, hi], "ic_contem_zero": bool(lo <= 0 <= hi)}


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = deduplicate(clean_static(load_raw()), verbose=False)
    X_tr, X_te, y_tr, y_te = split(df)
    y = y_te.to_numpy()
    n = len(y)
    bundle = joblib.load(MODEL_PATH)
    cands = joblib.load(CANDIDATES_PATH)
    cv_json = json.loads((METRICS_DIR / "cv_results.json").read_text(encoding="utf-8"))["models"]

    base = build_candidates()["lgbm"]
    # Contrafactual: mesma configuração da v0.2.0, mas COM idade (sem restrição monotônica nela).
    drop_com_idade = tuple(c for c in MODEL_EXCLUDED_FEATURES if c != "idade")
    com_idade = Pipeline([("pre", build_preprocessor("tree", drop_cols=drop_com_idade)), ("clf", build_lgbm(drop_com_idade))])

    M = {}
    for name, pipe in (("lgbm", base), ("lgbm_com_idade", com_idade)):
        oof, cal, final, aucs = oof_and_final(pipe, X_tr, y_tr)
        raw = final.predict_proba(X_te)[:, 1]
        pd_oof = cal.predict(oof)
        M[name] = {"raw": raw, "pd": cal.predict(raw), "pd_oof": pd_oof, "thr": cost_threshold(y_tr, pd_oof), "cv_auc": aucs}
    lr = cands["logreg"]
    raw = lr["pipeline"].predict_proba(X_te)[:, 1]
    M["logreg"] = {"raw": raw, "pd": lr["calibrator"].predict(raw), "thr": lr["thresholds"]["cost"],
                   "cv_auc": cv_json["logreg"]["auc"]["folds"]}

    # Consistência: o retreino aqui tem de reproduzir exatamente o modelo salvo.
    assert abs(M["lgbm"]["thr"] - bundle["threshold"]) < 1e-12, "corte diverge do model.joblib"
    assert np.allclose(M["lgbm"]["cv_auc"], cv_json["lgbm"]["auc"]["folds"], atol=1e-12), "AUC por fold diverge"

    # ---------- T8.1/T8.2 + Q5: bootstrap (mesmos índices para todos os modelos => pareado) ----------
    age = X_te["idade"].to_numpy()
    young, old = age < 30, age >= 60
    names = list(M)
    boot = {m: {"auc": [], "ks": [], "profit": [], "gap": []} for m in names}
    rng = np.random.default_rng(SEED)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, yo, yy = y[idx], old[idx], young[idx]
        for m in names:
            r, p = M[m]["raw"][idx], M[m]["pd"][idx]
            boot[m]["auc"].append(roc_auc_score(yb, r))
            boot[m]["ks"].append(ks_2samp(r[yb == 1], r[yb == 0]).statistic)
            boot[m]["profit"].append(profit_at(yb, p, M[m]["thr"]) / n)
            app = p < M[m]["thr"]
            # equal opportunity: TPR(60+) − TPR(<30) entre bons pagadores
            boot[m]["gap"].append(app[(yb == 0) & yo].mean() - app[(yb == 0) & yy].mean())
    boot = {m: {k: np.array(v) for k, v in d.items()} for m, d in boot.items()}

    def point(m):
        yv = y
        return {
            "auc": roc_auc_score(yv, M[m]["raw"]),
            "ks": ks_2samp(M[m]["raw"][yv == 1], M[m]["raw"][yv == 0]).statistic,
            "profit": profit_at(yv, M[m]["pd"], M[m]["thr"]) / n,
        }

    pt = {m: point(m) for m in names}
    models_ci = {m: {"auc": _summ(boot[m]["auc"], pt[m]["auc"]), "ks": _summ(boot[m]["ks"], pt[m]["ks"]),
                     "profit_per_proposal": _summ(boot[m]["profit"], pt[m]["profit"])} for m in names}

    def paired(a, b):
        return {k: _summ(boot[a][k] - boot[b][k], pt[a][k] - pt[b][k]) for k in ("auc", "ks", "profit")}

    paired_out = {"lgbm_menos_logreg": paired("lgbm", "logreg"), "lgbm_menos_lgbm_com_idade": paired("lgbm", "lgbm_com_idade")}

    folds = {m: M[m]["cv_auc"] for m in ("lgbm", "logreg", "lgbm_com_idade")}
    d = np.array(folds["lgbm"]) - np.array(folds["logreg"])
    cv_fold_out = {
        **{k: [float(x) for x in v] for k, v in folds.items()},
        "dif_lgbm_logreg": [float(x) for x in d],
        "dif_media": float(d.mean()), "dif_desvio": float(d.std(ddof=1)),
        "lgbm_vence_em_todos_os_folds": bool((d > 0).all()),
        "dif_lgbm_menos_com_idade_media": float((np.array(folds["lgbm"]) - np.array(folds["lgbm_com_idade"])).mean()),
    }

    # ---------- T8.3: ablação da idade (final sem idade × contrafactual com idade) ----------
    ok = ~np.isnan(age)
    abl = {}
    for m in ("lgbm", "lgbm_com_idade"):
        app = M[m]["pd"] < M[m]["thr"]
        t = group_table(X_te["idade"][ok], y[ok], app[ok])
        ps = portfolio_stats(y, M[m]["pd"], M[m]["thr"])
        abl[m] = {
            "corte_custo": float(M[m]["thr"]),
            "auc_teste": float(pt[m]["auc"]), "ks_teste": float(pt[m]["ks"]),
            "brier_calibrado": float(brier_score_loss(y, M[m]["pd"])),
            "lucro_por_proposta": float(pt[m]["profit"]),
            "taxa_aprovacao": ps["approval_rate"], "inadimplencia_aprovados": ps["default_rate_approved"],
            "disparate_impact_min": float(t["disparate_impact"].min()),
            "tpr_por_faixa": {g: float(v) for g, v in t["tpr_bom_aprovado"].items()},
            "aprovacao_por_faixa": {g: float(v) for g, v in t["taxa_aprovacao"].items()},
            "equal_opp_gap_60mais_menos_menor30": float(t.loc["60+", "tpr_bom_aprovado"] - t.loc["<30", "tpr_bom_aprovado"]),
            "equal_opp_gap_max_min": float(t["tpr_bom_aprovado"].max() - t["tpr_bom_aprovado"].min()),
            "gap_bootstrap": _summ(boot[m]["gap"], t.loc["60+", "tpr_bom_aprovado"] - t.loc["<30", "tpr_bom_aprovado"]),
        }
    # Positivo = tirar a idade (modelo final) reduziu a diferença de TPR em relação ao contrafactual com idade.
    gap_diff = boot["lgbm_com_idade"]["gap"] - boot["lgbm"]["gap"]
    abl["gap_reducao_bootstrap"] = _summ(gap_diff, abl["lgbm_com_idade"]["equal_opp_gap_60mais_menos_menor30"]
                                         - abl["lgbm"]["equal_opp_gap_60mais_menos_menor30"])

    # ---------- T8.4: sensibilidade da perda por mau aprovado (sem retreino) ----------
    pd_oof, pd_te = M["lgbm"]["pd_oof"], M["lgbm"]["pd"]
    t_base = M["lgbm"]["thr"]
    sens = []
    for loss in LOSSES:
        c = {**COST_MATRIX, "approve_bad": loss}
        t_opt = float(THRESHOLD_GRID[int(np.argmax(profit_curve(y_tr, pd_oof, c=c)))])
        aa = profit_at(y, np.zeros(n), 1.01, c) / n
        ra = profit_at(y, np.zeros(n), 0.0, c) / n
        lucro = profit_at(y, pd_te, t_opt, c) / n
        ps = portfolio_stats(y, pd_te, t_opt)
        sens.append({
            "perda_mau_aprovado": loss,
            "corte_teorico": theoretical_threshold(c),
            "corte_otimo_oof": t_opt,
            "lucro_teste_corte_otimo": lucro,
            "lucro_teste_aprovar_todos": aa,
            "lucro_teste_recusar_todos": ra,
            "ganho_pct_sobre_aprovar_todos": (lucro - aa) / abs(aa) * 100,
            "corte_base": float(t_base),
            "lucro_teste_mantendo_corte_base": profit_at(y, pd_te, t_base, c) / n,
            "lucro_teste_corte_0.5": profit_at(y, pd_te, 0.5, c) / n,
            "taxa_aprovacao": ps["approval_rate"],
            "inadimplencia_aprovados": ps["default_rate_approved"],
        })

    out = {"bootstrap": {"B": B, "seed": SEED, "n_teste": n, "modelos": models_ci, "pareado": paired_out},
           "auc_por_fold_cv": cv_fold_out, "ablacao_idade": abl, "sensibilidade_custo": sens,
           # E3: faixa REVISAR × parâmetros ilustrativos da revisão (corte e PD do modelo final, no teste)
           "sensibilidade_revisao": review_sensitivity(y, M["lgbm"]["pd"], M["lgbm"]["thr"])}
    (METRICS_DIR / "stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    f = lambda x: f"{x:.4f}"
    ci = lambda s: f"{f(s['est'])} [{f(s['ci95'][0])}; {f(s['ci95'][1])}]"
    print("== Bootstrap (B=%d) ==" % B)
    for m in names:
        print(f"{m:15s} AUC {ci(models_ci[m]['auc'])} | KS {ci(models_ci[m]['ks'])} | lucro {ci(models_ci[m]['profit_per_proposal'])}")
    for k, v in paired_out.items():
        print(k, {q: (ci(s), "contém 0" if s["ic_contem_zero"] else "NÃO contém 0") for q, s in v.items()})
    print("dif AUC por fold:", [round(x, 4) for x in cv_fold_out["dif_lgbm_logreg"]])
    print("== Ablação ==")
    print(json.dumps({k: v for k, v in abl.items()}, indent=1, ensure_ascii=False))
    print("== Sensibilidade ==")
    for s in sens:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print("== Sensibilidade da faixa REVISAR ==")
    for s in out["sensibilidade_revisao"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
