"""Métricas, corte por custo e avaliação final no TESTE (rodar uma única vez: python src/evaluate.py)."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import COST_MATRIX, FIGURES_DIR, METRICS_DIR, MODELS_DIR, MODEL_PATH, REVIEW_PARAMS, REVIEW_SENSITIVITY
from data import TARGET_PT, clean_static, deduplicate, load_raw, split

CANDIDATES_PATH = MODELS_DIR / "candidates.joblib"
THRESHOLD_GRID = np.round(np.arange(0.01, 0.9901, 0.005), 3)

COLORS = {"dummy": "#8a8a8a", "logreg": "#e08a1e", "lgbm": "#2a6f97"}
LABELS = {"dummy": "Dummy (prior)", "logreg": "Regressão logística", "lgbm": "LightGBM"}
CALIBRATION_LABELS = {"none": "sem calibrador", "sigmoid": "Platt (sigmoide)", "isotonic": "isotônica"}


def compute_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_2samp(p[y == 1], p[y == 0]).statistic,
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def theoretical_threshold(c=COST_MATRIX) -> float:
    """Aprovar se (1-p)*ag + p*ab > (1-p)*rg + p*rb  =>  p < (ag-rg) / ((ag-rg) + (rb-ab))."""
    gain_good = c["approve_good"] - c["reject_good"]
    gain_bad = c["reject_bad"] - c["approve_bad"]
    return gain_good / (gain_good + gain_bad)


def profit_curve(y, pd_, grid=THRESHOLD_GRID, c=COST_MATRIX) -> np.ndarray:
    """Lucro TOTAL para cada corte t: aprova se PD < t."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    approve = pd_[:, None] < grid[None, :]
    good = (y == 0)[:, None]
    bad = (y == 1)[:, None]
    return (
        (approve & good).sum(0) * c["approve_good"]
        + (approve & bad).sum(0) * c["approve_bad"]
        + (~approve & good).sum(0) * c["reject_good"]
        + (~approve & bad).sum(0) * c["reject_bad"]
    )


def profit_at(y, pd_, t, c=COST_MATRIX) -> float:
    y, pd_ = np.asarray(y), np.asarray(pd_)
    app = pd_ < t
    return float(
        ((app) & (y == 0)).sum() * c["approve_good"]
        + ((app) & (y == 1)).sum() * c["approve_bad"]
        + ((~app) & (y == 0)).sum() * c["reject_good"]
        + ((~app) & (y == 1)).sum() * c["reject_bad"]
    )


def cost_threshold(y, pd_) -> float:
    return float(THRESHOLD_GRID[int(np.argmax(profit_curve(y, pd_)))])


def f1_threshold(y, pd_) -> float:
    """Corte que maximiza o F1 da classe 'mau' (prevê mau se PD >= t)."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    pred = pd_[:, None] >= THRESHOLD_GRID[None, :]
    tp = (pred & (y == 1)[:, None]).sum(0)
    fp = (pred & (y == 0)[:, None]).sum(0)
    fn = ((~pred) & (y == 1)[:, None]).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(THRESHOLD_GRID[int(np.argmax(f1))])


def review_band(c=COST_MATRIX, params=REVIEW_PARAMS):
    """Faixa de PD [lo, hi] em que revisar tem valor esperado maior que decidir automaticamente (ADR-20).
    Revisor: aprova a fração `especificidade_analista` dos bons e recusa a fração `sensibilidade_analista` dos maus,
    a um custo fixo `custo_revisao` por caso. Valor de revisar com PD p:
        V(p) = (1-p)*g + p*b - custo,  g = e*ag + (1-e)*rg,  b = s*rb + (1-s)*ab
    Revisar > aprovar  <=>  p > lo;  revisar > recusar  <=>  p < hi. Devolve None se a revisão nunca compensa."""
    s, e, k = params["sensibilidade_analista"], params["especificidade_analista"], params["custo_revisao"]
    ag, ab, rg, rb = c["approve_good"], c["approve_bad"], c["reject_good"], c["reject_bad"]
    g = e * ag + (1 - e) * rg
    b = s * rb + (1 - s) * ab
    den_lo = (b - ab) + (ag - g)
    den_hi = (g - rg) + (rb - b)
    if den_lo <= 0 or den_hi <= 0:
        return None
    lo = max(0.0, (ag - g + k) / den_lo)
    hi = min(1.0, (g - rg - k) / den_hi)
    return (float(lo), float(hi)) if lo < hi else None


def band_or_cut(band, t):
    """Faixa vazia -> decisão em 2 faixas no corte t (lo = hi = t)."""
    return (float(band[0]), float(band[1])) if band else (float(t), float(t))


def portfolio_3faixas(y, pd_, band, t, c=COST_MATRIX, params=REVIEW_PARAMS) -> dict:
    """Decisão em 3 faixas: volumes, taxa de mau por faixa e lucro ESPERADO com o revisor ilustrativo."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    lo, hi = band_or_cut(band, t)
    apr, rec = pd_ < lo, pd_ > hi
    rev = ~apr & ~rec
    por = {}
    for nome, m in (("APROVAR", apr), ("REVISAR", rev), ("RECUSAR", rec)):
        por[nome] = {"n": int(m.sum()), "pct": float(m.mean()), "taxa_mau": float(y[m].mean()) if m.any() else None,
                     "maus": int(y[m].sum()), "bons": int((1 - y[m]).sum())}
    s, e, k = params["sensibilidade_analista"], params["especificidade_analista"], params["custo_revisao"]
    good, bad = y == 0, y == 1
    auto = ((apr & good).sum() * c["approve_good"] + (apr & bad).sum() * c["approve_bad"]
            + (rec & good).sum() * c["reject_good"] + (rec & bad).sum() * c["reject_bad"])
    n_rev_good, n_rev_bad = int((rev & good).sum()), int((rev & bad).sum())
    revisao = (n_rev_good * (e * c["approve_good"] + (1 - e) * c["reject_good"])
               + n_rev_bad * (s * c["reject_bad"] + (1 - s) * c["approve_bad"]) - k * int(rev.sum()))
    n = len(y)
    return {
        "faixa": [lo, hi],
        "faixa_vazia": band is None,
        "parametros_revisao": dict(params),
        "por_faixa": por,
        "aprovacao_automatica": float(apr.mean()),
        "inadimplencia_aprovados_automaticos": float(y[apr].mean()) if apr.any() else float("nan"),
        "lucro_esperado_por_proposta": float((auto + revisao) / n),
        "referencias_por_proposta": {
            "revisar_todos_aprovados": float((auto + n_rev_good * c["approve_good"] + n_rev_bad * c["approve_bad"]) / n),
            "revisar_todos_recusados": float((auto + n_rev_good * c["reject_good"] + n_rev_bad * c["reject_bad"]) / n),
            "revisor_perfeito_sem_custo": float((auto + n_rev_good * c["approve_good"] + n_rev_bad * c["reject_bad"]) / n),
        },
    }


def review_sensitivity(y, pd_, t, grid=REVIEW_SENSITIVITY, c=COST_MATRIX) -> list:
    """Faixa, volume e lucro esperado para cada combinação dos parâmetros ilustrativos da revisão."""
    rows = []
    for k in grid["custo_revisao"]:
        for s in grid["sensibilidade_analista"]:
            for e in grid["especificidade_analista"]:
                p = {"custo_revisao": k, "sensibilidade_analista": s, "especificidade_analista": e}
                band = review_band(c, p)
                r = portfolio_3faixas(y, pd_, band, t, c, p)
                rows.append({**p, "faixa": r["faixa"] if band else None, "pct_revisar": r["por_faixa"]["REVISAR"]["pct"],
                             "taxa_mau_revisar": r["por_faixa"]["REVISAR"]["taxa_mau"],
                             "aprovacao_automatica": r["aprovacao_automatica"],
                             "lucro_esperado_por_proposta": r["lucro_esperado_por_proposta"]})
    return rows


def portfolio_stats(y, pd_, t) -> dict:
    y, pd_ = np.asarray(y), np.asarray(pd_)
    app = pd_ < t
    n_app = int(app.sum())
    return {
        "threshold": float(t),
        "approval_rate": float(app.mean()),
        "default_rate_approved": float(y[app].mean()) if n_app else float("nan"),
        "profit_total": profit_at(y, pd_, t),
        "profit_per_proposal": profit_at(y, pd_, t) / len(y),
        "tp_recusa_mau": int(((~app) & (y == 1)).sum()),
        "fp_recusa_bom": int(((~app) & (y == 0)).sum()),
        "fn_aprova_mau": int((app & (y == 1)).sum()),
        "tn_aprova_bom": int((app & (y == 0)).sum()),
    }


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def _fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = deduplicate(clean_static(load_raw()), verbose=False)
    _, X_te, _, y_te = split(df)  # único ponto de uso do teste
    y = y_te.to_numpy()

    cands = joblib.load(CANDIDATES_PATH)
    bundle = joblib.load(MODEL_PATH)
    best = bundle["best_name"]

    raw, cal, results = {}, {}, {}
    for name, c in cands.items():
        raw[name] = c["pipeline"].predict_proba(X_te)[:, 1]
        cal[name] = c["calibrator"].predict(raw[name])
        m_raw, m_cal = compute_metrics(y, raw[name]), compute_metrics(y, cal[name])
        t_cost = c["thresholds"]["cost"]
        results[name] = {
            "auc": m_raw["auc"],
            "gini": m_raw["gini"],
            "ks": m_raw["ks"],
            "pr_auc": m_raw["pr_auc"],
            "brier_raw": m_raw["brier"],
            "brier_calibrated": m_cal["brier"],
            "auc_calibrated": m_cal["auc"],
            "ks_calibrated": m_cal["ks"],
            "at_cost_threshold": portfolio_stats(y, cal[name], t_cost),
        }

    # Q2: três regras de corte (escolhidas no OOF do treino) aplicadas ao teste, modelo final calibrado
    th = bundle["thresholds"]
    rules = {
        "padrao_0.5": th["default_0.5"],
        "f1": th["f1"],
        "custo": th["cost"],
        "teorico_0.286": th["theoretical"],
    }
    rules_test = {k: portfolio_stats(y, cal[best], t) for k, t in rules.items()}
    tres_faixas = portfolio_3faixas(y, cal[best], bundle.get("review_band"), th["cost"],
                                    params=bundle.get("review_params", REVIEW_PARAMS))
    always_approve = profit_at(y, np.zeros_like(y, dtype=float), 1.01)
    always_reject = profit_at(y, np.zeros_like(y, dtype=float), 0.0)

    out = {
        "n_test": int(len(y)),
        "test_default_rate": float(y.mean()),
        "best_model": best,
        "models": results,
        "cut_rules_final_model": rules_test,
        "decisao_3_faixas": tres_faixas,
        "baselines_policy": {
            "aprovar_todos_profit_per_proposal": always_approve / len(y),
            "recusar_todos_profit_per_proposal": always_reject / len(y),
        },
        "cost_matrix": COST_MATRIX,
        "model_version": bundle["model_version"],
        "data_sha256": bundle["data_sha256"],
    }
    (METRICS_DIR / "test_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # tabela .md
    L = ["| Modelo | AUC | Gini | KS | PR-AUC | Brier (cru) | Brier (calibrado) |", "|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        L.append(
            f"| {LABELS[n]} | {_fmt(r['auc'])} | {_fmt(r['gini'])} | {_fmt(r['ks'])} | {_fmt(r['pr_auc'])} "
            f"| {_fmt(r['brier_raw'])} | {_fmt(r['brier_calibrated'])} |"
        )
    L += ["", f"Modelo final: **{LABELS[best]}** (calibrado). Teste: n={len(y)}, inadimplência {y.mean():.2%}.", "",
          "| Regra de corte | Corte | Lucro/proposta | Aprovação | Inadimplência aprovados |", "|---|---|---|---|---|"]
    for k, s in rules_test.items():
        L.append(
            f"| {k} | {s['threshold']:.3f} | {s['profit_per_proposal']:.4f} | {s['approval_rate']:.2%} "
            f"| {s['default_rate_approved']:.2%} |"
        )
    lo3, hi3 = tres_faixas["faixa"]
    L += ["", f"Decisão em 3 faixas (produto): REVISAR se {lo3:.3f} ≤ PD ≤ {hi3:.3f}"
              f"{' (faixa vazia: decisão em 2 faixas)' if tres_faixas['faixa_vazia'] else ''}. "
              f"Lucro esperado por proposta com o revisor ilustrativo: {tres_faixas['lucro_esperado_por_proposta']:.4f}.", "",
          "| Faixa | Propostas | % | Taxa de mau |", "|---|---|---|---|"]
    for k, s in tres_faixas["por_faixa"].items():
        taxa = "—" if s["taxa_mau"] is None else f"{s['taxa_mau']:.2%}"
        L.append(f"| {k} | {s['n']} | {s['pct']:.2%} | {taxa} |")
    (METRICS_DIR / "test_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- figuras ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in raw:
        fpr, tpr, _ = roc_curve(y, raw[n])
        ax.plot(fpr, tpr, color=COLORS[n], lw=2, label=f"{LABELS[n]} (AUC {results[n]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC (teste)")
    ax.legend(loc="lower right")
    _save(fig, "roc.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in raw:
        pr, rc, _ = precision_recall_curve(y, raw[n])
        ax.plot(rc, pr, color=COLORS[n], lw=2, label=f"{LABELS[n]} (PR-AUC {results[n]['pr_auc']:.3f})")
    ax.axhline(y.mean(), ls=":", color="gray")
    ax.text(0.98, y.mean() + 0.01, "base = inadimplência", ha="right", color="gray")
    ax.set(xlabel="Recall", ylabel="Precisão", title="Curva Precisão × Recall (teste)")
    ax.legend(loc="upper right")
    _save(fig, "pr.png")

    # KS do modelo final (calibrado): distância máxima entre as ECDFs de bons e maus
    s = cal[best]
    xs = np.sort(np.unique(s))
    cdf_good = np.searchsorted(np.sort(s[y == 0]), xs, side="right") / (y == 0).sum()
    cdf_bad = np.searchsorted(np.sort(s[y == 1]), xs, side="right") / (y == 1).sum()
    k = int(np.argmax(np.abs(cdf_good - cdf_bad)))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(xs, cdf_good, color="#2a6f97", lw=2, label="Bons pagadores")
    ax.plot(xs, cdf_bad, color="#c1443c", lw=2, label="Maus pagadores")
    ax.vlines(xs[k], cdf_bad[k], cdf_good[k], color="black", ls="--")
    ax.text(xs[k] + 0.01, (cdf_good[k] + cdf_bad[k]) / 2, f"KS = {abs(cdf_good[k]-cdf_bad[k]):.3f}")
    ax.set(xlabel="PD calibrada", ylabel="Distribuição acumulada", title=f"KS · {LABELS[best]} (teste)")
    ax.legend(loc="lower right")
    _save(fig, "ks.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    cal_lab = f"Depois ({CALIBRATION_LABELS.get(bundle.get('calibration_method', 'isotonic'), 'calibrado')})"
    for arr, lab, col in ((raw[best], "Antes (probabilidade crua)", "#c1443c"), (cal[best], cal_lab, "#2a6f97")):
        frac, mean_p = calibration_curve(y, arr, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=col, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], ls=":", color="gray", label="Calibração perfeita")
    ax.set(xlabel="PD prevista (média do decil)", ylabel="Taxa observada de maus",
           title=f"Calibração · {LABELS[best]} (teste)")
    ax.legend(loc="upper left")
    _save(fig, "calibration.png")

    # curva de lucro com as 4 marcações
    per = profit_curve(y, cal[best]) / len(y)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(THRESHOLD_GRID, per, color="#2a6f97", lw=2)
    marks = {"0,5": ("padrao_0.5", "#8a8a8a"), "F1": ("f1", "#e08a1e"),
             "custo": ("custo", "#2a8f4f"), "teórico 0,286": ("teorico_0.286", "#c1443c")}
    for i, (lab, (key, col)) in enumerate(marks.items()):
        t = rules[key]
        v = rules_test[key]["profit_per_proposal"]
        ax.axvline(t, color=col, ls="--", lw=1.2)
        ax.plot(t, v, "o", color=col)
        ax.text(0.97, 0.34 - 0.09 * i, f"{lab}: corte {t:.3f} · lucro {v:.3f}", color=col, fontsize=9,
                ha="right", transform=ax.transAxes)
    ax.set(xlabel="Corte (aprova se PD < corte)", ylabel="Lucro esperado por proposta",
           title="Lucro × corte (teste, premissas ilustrativas)")
    _save(fig, "profit_curve.png")

    s = rules_test["custo"]
    cm = np.array([[s["tn_aprova_bom"], s["fn_aprova_mau"]], [s["fp_recusa_bom"], s["tp_recusa_mau"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}".replace(",", "."), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Real: bom", "Real: mau"])
    ax.set_yticks([0, 1], ["Aprovado", "Recusado"])
    ax.set_title(f"Matriz de confusão · corte {s['threshold']:.3f} (teste)")
    _save(fig, "confusion.png")

    # --- pontos das curvas para o app (regra de ouro: o app só lê, nunca recalcula) ---
    def thin(*arrs, n=250):
        idx = np.unique(np.linspace(0, len(arrs[0]) - 1, min(n, len(arrs[0]))).astype(int))
        return [np.round(np.asarray(a)[idx], 5).tolist() for a in arrs]

    curves = {"model": best, "calibration_method": bundle.get("calibration_method", "isotonic"),
              "roc": {}, "pr": {}, "calibration": {}, "profit": {}, "ks": {}}
    for n in raw:
        fpr, tpr, _ = roc_curve(y, raw[n])
        curves["roc"][n] = dict(zip(("fpr", "tpr"), thin(fpr, tpr)), auc=float(results[n]["auc"]))
        pr, rc, _ = precision_recall_curve(y, raw[n])
        curves["pr"][n] = dict(zip(("recall", "precision"), thin(rc, pr)), pr_auc=float(results[n]["pr_auc"]))
    curves["pr"]["baseline"] = float(y.mean())
    xs_t, g_t, b_t = thin(xs, cdf_good, cdf_bad)
    curves["ks"] = {"x": xs_t, "cdf_good": g_t, "cdf_bad": b_t, "ks": float(abs(cdf_good[k] - cdf_bad[k])),
                    "x_ks": float(xs[k])}
    for key, arr in (("before", raw[best]), ("after", cal[best])):
        frac, mean_p = calibration_curve(y, arr, n_bins=10, strategy="quantile")
        curves["calibration"][key] = {"mean_pred": np.round(mean_p, 5).tolist(), "frac_pos": np.round(frac, 5).tolist()}
    curves["profit"] = {
        "thresholds": THRESHOLD_GRID.tolist(),
        "profit_per_proposal": np.round(per, 5).tolist(),
        "marks": {k_: {"corte": float(rules[k_]), "lucro": float(rules_test[k_]["profit_per_proposal"])} for k_ in rules},
    }
    (METRICS_DIR / "curves.json").write_text(json.dumps(curves, ensure_ascii=False), encoding="utf-8")

    print(open(METRICS_DIR / "test_results.md", encoding="utf-8").read())
    print(f"Referência: aprovar todos = {always_approve/len(y):.4f} | recusar todos = {always_reject/len(y):.4f} (lucro/proposta)")


if __name__ == "__main__":
    main()
