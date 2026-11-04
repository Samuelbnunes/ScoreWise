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

from config import COST_MATRIX, FIGURES_DIR, METRICS_DIR, MODELS_DIR, MODEL_PATH
from data import TARGET_PT, clean_static, deduplicate, load_raw, split

CANDIDATES_PATH = MODELS_DIR / "candidates.joblib"
THRESHOLD_GRID = np.round(np.arange(0.01, 0.9901, 0.005), 3)

COLORS = {"dummy": "#8a8a8a", "logreg": "#e08a1e", "lgbm": "#2a6f97"}
LABELS = {"dummy": "Dummy (prior)", "logreg": "Regressão logística", "lgbm": "LightGBM"}


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
    always_approve = profit_at(y, np.zeros_like(y, dtype=float), 1.01)
    always_reject = profit_at(y, np.zeros_like(y, dtype=float), 0.0)

    out = {
        "n_test": int(len(y)),
        "test_default_rate": float(y.mean()),
        "best_model": best,
        "models": results,
        "cut_rules_final_model": rules_test,
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
    for arr, lab, col in ((raw[best], "Antes (probabilidade crua)", "#c1443c"), (cal[best], "Depois (isotônica)", "#2a6f97")):
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

    print(open(METRICS_DIR / "test_results.md", encoding="utf-8").read())
    print(f"Referência: aprovar todos = {always_approve/len(y):.4f} | recusar todos = {always_reject/len(y):.4f} (lucro/proposta)")


if __name__ == "__main__":
    main()
