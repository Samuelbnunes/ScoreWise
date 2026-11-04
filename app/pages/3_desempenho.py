"""Desempenho: KPIs, curvas interativas (curves.json), tabela com IC e sensibilidade de custos (stats.json)."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
for _p in (_APP, _APP.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from ui import charts
from ui.components import begin, br, ci_text, end, kpi_card, pct, require_metrics, section_header

begin("Desempenho")
m = require_metrics("test_results.json", "curves.json", "stats.json")
test, curves, stats = m["test_results.json"], m["curves.json"], m["stats.json"]
lg, lr = test["models"]["lgbm"], test["models"]["logreg"]
boot = stats["bootstrap"]

section_header("Desempenho no conjunto de teste", f"{br(test['n_test'], 0)} propostas, usadas uma única vez · inadimplência de {pct(test['test_default_rate'], 2)}")
k = st.columns(4)
with k[0]:
    kpi_card("AUC", br(lg["auc"], 3), "ordena maus acima de bons", "Probabilidade de o modelo ordenar um mau pagador acima de um bom.")
with k[1]:
    kpi_card("Gini", br(lg["gini"], 3), "2 × AUC − 1", "Reescala da AUC: 0 = acaso, 1 = perfeito.")
with k[2]:
    kpi_card("KS", br(lg["ks"], 3), "separação máxima", "Maior distância entre as distribuições acumuladas de bons e maus.")
with k[3]:
    kpi_card("Brier calibrado", br(lg["brier_calibrated"], 4), f"antes da calibração: {br(lg['brier_raw'], 4)}",
             "Erro quadrático médio da probabilidade. Menor é melhor; a calibração corrige o viés do class_weight.")

st.write("")
tabs = st.tabs(["Curva ROC", "Precisão × Recall", "KS", "Calibração", "Lucro × corte"])
with tabs[0], st.container(border=True):
    st.plotly_chart(charts.roc(curves), width="stretch", config=charts.CFG)
    st.caption("Quanto mais a curva se aproxima do canto superior esquerdo, melhor a separação entre bons e maus pagadores.")
with tabs[1], st.container(border=True):
    st.plotly_chart(charts.pr(curves), width="stretch", config=charts.CFG)
    st.caption("Com classe rara (poucos maus), a curva PR é mais exigente que a ROC; a linha pontilhada é o desempenho de um palpite aleatório.")
with tabs[2], st.container(border=True):
    st.plotly_chart(charts.ks(curves), width="stretch", config=charts.CFG)
    st.caption("Distância máxima entre as distribuições acumuladas de bons e maus pagadores, pelo score do modelo calibrado.")
with tabs[3], st.container(border=True):
    st.plotly_chart(charts.calibration(curves), width="stretch", config=charts.CFG)
    st.caption(f"Antes da calibração o modelo superestima a probabilidade (Brier {br(lg['brier_raw'], 4)}); depois, a curva cola na diagonal (Brier {br(lg['brier_calibrated'], 4)}).")
with tabs[4], st.container(border=True):
    st.plotly_chart(charts.profit(curves), width="stretch", config=charts.CFG)
    r = test["cut_rules_final_model"]
    st.caption(f"Premissas ilustrativas. O corte por custo ({br(r['custo']['threshold'], 3)}) rende {br(r['custo']['profit_per_proposal'], 4)} por proposta, contra "
               f"{br(r['padrao_0.5']['profit_per_proposal'], 4)} no corte 0,5 e {br(test['baselines_policy']['aprovar_todos_profit_per_proposal'], 4)} ao aprovar todos.")

section_header("Comparativo de modelos", "Teste, com IC 95% por bootstrap pareado (1.000 reamostragens)")
mc = boot["modelos"]
rows = [{"Modelo": "Dummy (piso)", "AUC [IC 95%]": br(test["models"]["dummy"]["auc"], 3), "Gini": br(test["models"]["dummy"]["gini"], 3),
         "KS": br(test["models"]["dummy"]["ks"], 3), "PR-AUC": br(test["models"]["dummy"]["pr_auc"], 3), "Brier calibrado": br(test["models"]["dummy"]["brier_calibrated"], 4),
         "Lucro/proposta": "—"}]
for key, label in (("logreg", "Regressão logística"), ("lgbm", "LightGBM (final)")):
    t, c = test["models"][key], mc[key]
    rows.append({"Modelo": label, "AUC [IC 95%]": ci_text(c["auc"]["est"], *c["auc"]["ci95"]), "Gini": br(t["gini"], 3), "KS": br(t["ks"], 3),
                 "PR-AUC": br(t["pr_auc"], 3), "Brier calibrado": br(t["brier_calibrated"], 4),
                 "Lucro/proposta": ci_text(c["profit_per_proposal"]["est"], *c["profit_per_proposal"]["ci95"], nd=4)})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
d = boot["pareado"]["lgbm_menos_logreg"]["auc"]
sig = "não contém 0" if not d["ic_contem_zero"] else "contém 0"
st.markdown(f"**LightGBM − logística (ΔAUC pareado):** {ci_text(d['est'], *d['ci95'], nd=4)}. O IC {sig}: a melhora é "
            f"{'estatisticamente significativa a 95%, porém pequena' if not d['ic_contem_zero'] else 'não significativa a 95%'}.")

section_header("Sensibilidade às premissas de custo", "Só a perda por mau aprovado varia (+1 / −1 / 0 fixos). Sem retreino: o corte é recalculado nas predições out-of-fold do treino")
sens = pd.DataFrame([{
    "Perda por mau aprovado": f"−{abs(int(s['perda_mau_aprovado']))}", "Corte teórico": br(s["corte_teorico"], 3), "Corte ótimo (OOF)": br(s["corte_otimo_oof"], 3),
    "Lucro/proposta": br(s["lucro_teste_corte_otimo"], 4), "Aprovar todos": br(s["lucro_teste_aprovar_todos"], 4),
    "Ganho vs. aprovar todos": f"+{br(s['ganho_pct_sobre_aprovar_todos'], 1)}%", "Mantendo corte 0,270": br(s["lucro_teste_mantendo_corte_0.270"], 4),
    "Corte 0,5": br(s["lucro_teste_corte_0.5"], 4),
} for s in stats["sensibilidade_custo"]])
st.dataframe(sens, hide_index=True, width="stretch")
st.caption("O valor do modelo depende fortemente da premissa de custo: o corte deve ser recalibrado quando a economia do produto muda.")

end(test["model_version"])
