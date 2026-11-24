"""Equidade por faixa etária no TESTE, com a decisão do modelo final (aprova se PD < corte).
'Positivo' para equal opportunity = bom pagador APROVADO (TPR = P(aprovado | bom)).
v0.2.0: a idade não é variável do modelo (ADR-18); ela é usada aqui só para auditar o resultado. Além do corte
simples, mede a decisão em 3 faixas do produto (REVISAR conta como "não aprovado automaticamente")."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AGE_BINS, FIGURES_DIR, METRICS_DIR, MODEL_PATH
from data import clean_static, deduplicate, load_raw, split

AGE_LABELS = ["<30", "30–44", "45–59", "60+"]
DI_ALERT = 0.8  # regra dos 4/5


def group_table(idade, y, approved) -> pd.DataFrame:
    faixa = pd.cut(idade, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    rows = []
    for g in AGE_LABELS:
        m = (faixa == g).to_numpy()
        yy, aa = y[m], approved[m]
        rows.append(
            {
                "faixa": g,
                "n": int(m.sum()),
                "taxa_mau_real": float(yy.mean()),
                "taxa_aprovacao": float(aa.mean()),
                "tpr_bom_aprovado": float(aa[yy == 0].mean()),  # P(aprovado | bom)
                "fpr_mau_aprovado": float(aa[yy == 1].mean()),  # P(aprovado | mau)
                "inadimplencia_aprovados": float(yy[aa].mean()) if aa.any() else float("nan"),
            }
        )
    t = pd.DataFrame(rows).set_index("faixa")
    ref = t["taxa_aprovacao"].idxmax()
    t["disparate_impact"] = t["taxa_aprovacao"] / t.loc[ref, "taxa_aprovacao"]
    t["alerta_di_lt_0.8"] = t["disparate_impact"] < DI_ALERT
    t["equal_opp_diff_vs_ref"] = t["tpr_bom_aprovado"] - t.loc[ref, "tpr_bom_aprovado"]
    t.attrs["ref"] = ref
    return t


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODEL_PATH)
    df = deduplicate(clean_static(load_raw()), verbose=False)
    _, X_te, _, y_te = split(df)

    pd_te = bundle["calibrator"].predict(bundle["raw_pipeline"].predict_proba(X_te)[:, 1])
    approved = pd_te < bundle["threshold"]
    ok = X_te["idade"].notna().to_numpy()  # idade=0 virou NaN; fica fora da análise por faixa
    t = group_table(X_te["idade"][ok], y_te.to_numpy()[ok], approved[ok])

    eo_gap = float(t["tpr_bom_aprovado"].max() - t["tpr_bom_aprovado"].min())

    # Decisão em 3 faixas por faixa etária (ADR-20)
    from evaluate import band_or_cut

    lo, hi = band_or_cut(bundle.get("review_band"), bundle["threshold"])
    yk, pk = y_te.to_numpy()[ok], pd_te[ok]
    faixa = pd.cut(X_te["idade"][ok], bins=AGE_BINS, labels=AGE_LABELS, right=False).to_numpy()
    tres = []
    for g in AGE_LABELS:
        m = faixa == g
        auto, rev = pk[m] < lo, (pk[m] >= lo) & (pk[m] <= hi)
        tres.append({"faixa": g, "n": int(m.sum()), "aprovacao_automatica": float(auto.mean()), "revisar": float(rev.mean()),
                     "recusar": float((pk[m] > hi).mean()), "tpr_bom_aprovado_automatico": float(auto[yk[m] == 0].mean())})
    ref3 = max(tres, key=lambda r: r["aprovacao_automatica"])
    for r in tres:
        r["disparate_impact_automatico"] = r["aprovacao_automatica"] / ref3["aprovacao_automatica"]
    out = {
        "corte": float(bundle["threshold"]),
        "n_sem_idade_excluidos": int((~ok).sum()),
        "grupo_referencia": t.attrs["ref"],
        "por_faixa": t.reset_index().to_dict(orient="records"),
        "disparate_impact_min": float(t["disparate_impact"].min()),
        "equal_opportunity_gap_max": eo_gap,
        "aprovacao_global": float(approved.mean()),
        "faixa_revisar": [lo, hi],
        "decisao_3_faixas_por_faixa_etaria": tres,
        "decisao_3_faixas_grupo_referencia": ref3["faixa"],
        "decisao_3_faixas_disparate_impact_min": float(min(r["disparate_impact_automatico"] for r in tres)),
        "decisao_3_faixas_gap_tpr_max": float(max(r["tpr_bom_aprovado_automatico"] for r in tres)
                                               - min(r["tpr_bom_aprovado_automatico"] for r in tres)),
    }
    (METRICS_DIR / "fairness.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cores = ["#c1443c" if a else "#2a6f97" for a in t["alerta_di_lt_0.8"]]
    axes[0].bar(t.index, t["taxa_aprovacao"] * 100, color=cores)
    for i, v in enumerate(t["taxa_aprovacao"] * 100):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[0].set(title="Taxa de aprovação (DI em vermelho se < 0,8)", ylabel="%", ylim=(0, 110))
    axes[1].bar(t.index, t["taxa_mau_real"] * 100, color="#8a8a8a")
    for i, v in enumerate(t["taxa_mau_real"] * 100):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[1].set(title="Inadimplência real (risco observado)", ylabel="%")
    axes[2].bar(t.index, t["tpr_bom_aprovado"] * 100, color="#2a6f97")
    for i, v in enumerate(t["tpr_bom_aprovado"] * 100):
        axes[2].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[2].set(title="Bons pagadores aprovados (TPR)", ylabel="%", ylim=(0, 110))
    for ax, n in zip(axes, [None] * 3):
        ax.set_xticks(range(len(t)), [f"{g}\n(n={t.loc[g, 'n']})" for g in t.index])
    fig.suptitle(f"Equidade por faixa etária (teste, corte {bundle['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fairness_age.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    print(f"referência: {t.attrs['ref']} | DI mínimo: {t['disparate_impact'].min():.3f} | gap TPR: {eo_gap:.4f} | sem idade: {int((~ok).sum())}")
    print(f"3 faixas (REVISAR em [{lo:.3f}; {hi:.3f}]):")
    print(pd.DataFrame(tres).set_index("faixa").round(4).to_string())


if __name__ == "__main__":
    main()
