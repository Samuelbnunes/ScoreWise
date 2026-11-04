"""Equidade por faixa etária (fairness.json) e ablação sem idade (stats.json)."""
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

begin("Equidade")
m = require_metrics("fairness.json", "stats.json")
fj, stats = m["fairness.json"], m["stats.json"]
rows = fj["por_faixa"]
ab = stats["ablacao_sem_idade"]
ref = fj["grupo_referencia"]

section_header("Equidade por faixa etária", f"Teste, corte {br(fj['corte'], 3)} · grupo de referência: {ref} · só a idade pode ser analisada neste dataset")
k = st.columns(3)
with k[0]:
    kpi_card("Disparate impact mínimo", br(fj["disparate_impact_min"], 3), "limite de alerta: 0,8",
             "Razão entre a taxa de aprovação do grupo e a do grupo de maior aprovação. A regra dos 4/5 (0,8) é uma heurística de origem estadunidense, não norma brasileira.")
with k[1]:
    kpi_card("Diferença de TPR", f"{br(fj['equal_opportunity_gap_max'] * 100, 1)} p.p.", "bons pagadores aprovados, entre faixas",
             "Equal opportunity: diferença entre faixas na taxa de aprovação de quem é bom pagador.")
with k[2]:
    kpi_card("Aprovação global", pct(fj["aprovacao_global"]), "no corte por custo")

st.write("")
a, b = st.columns([1.4, 1], gap="large")
with a, st.container(border=True):
    st.plotly_chart(charts.fairness_bars(rows), width="stretch", config=charts.CFG)
with b, st.container(border=True):
    st.plotly_chart(charts.di_bars(rows), width="stretch", config=charts.CFG)

st.dataframe(pd.DataFrame([{
    "Faixa": r["faixa"], "n": br(r["n"], 0), "Inadimplência real": pct(r["taxa_mau_real"]), "Aprovação": pct(r["taxa_aprovacao"]),
    "Disparate impact": br(r["disparate_impact"], 3), "Alerta (DI < 0,8)": "Sim" if r["alerta_di_lt_0.8"] else "Não",
    "Bons aprovados (TPR)": pct(r["tpr_bom_aprovado"]), "Dif. TPR vs. ref.": f"{br(r['equal_opp_diff_vs_ref'] * 100, 1)} p.p.",
} for r in rows]), hide_index=True, width="stretch")
g = ab["lgbm"]["gap_bootstrap"]
st.markdown(
    f"**Leitura:** a diferença de aprovação é em boa parte explicada pelo risco observado, mas a *equal opportunity* mostra uma disparidade além dele: "
    f"entre os bons pagadores, a diferença de aprovação entre 60+ e <30 é de **{br(g['est'] * 100, 1)} p.p.** (IC 95%: {br(g['ci95'][0] * 100, 1)} a {br(g['ci95'][1] * 100, 1)}). "
    f"A faixa <30 tem só {br(rows[0]['n'], 0)} propostas, então suas taxas têm mais incerteza."
)

section_header("E se retirarmos a idade do modelo?", "Ablação (Q5): mesmo protocolo de treino, corte por custo recalculado, IC 95% por bootstrap")
w, s = ab["lgbm"], ab["lgbm_sem_idade"]
red = ab["gap_reducao_bootstrap"]
pair = stats["bootstrap"]["pareado"]["lgbm_menos_lgbm_sem_idade"]
tab = pd.DataFrame([
    {"Métrica": "AUC", "Com idade": br(w["auc_teste"], 4), "Sem idade": br(s["auc_teste"], 4), "Diferença [IC 95%]": ci_text(pair["auc"]["est"], *pair["auc"]["ci95"], nd=4)},
    {"Métrica": "Lucro por proposta", "Com idade": br(w["lucro_por_proposta"], 4), "Sem idade": br(s["lucro_por_proposta"], 4),
     "Diferença [IC 95%]": ci_text(pair["profit"]["est"], *pair["profit"]["ci95"], nd=4)},
    {"Métrica": "Disparate impact mínimo", "Com idade": br(w["disparate_impact_min"], 3), "Sem idade": br(s["disparate_impact_min"], 3), "Diferença [IC 95%]": "—"},
    {"Métrica": "Diferença de TPR (60+ − <30), p.p.", "Com idade": br(w["equal_opp_gap_60mais_menos_menor30"] * 100, 2),
     "Sem idade": br(s["equal_opp_gap_60mais_menos_menor30"] * 100, 2),
     "Diferença [IC 95%]": ci_text(red["est"] * 100, red["ci95"][0] * 100, red["ci95"][1] * 100, nd=2)},
])
st.dataframe(tab, hide_index=True, width="stretch")
st.info(
    f"Retirar a idade custa pouco AUC e não gera perda detectável de lucro, mas reduz a diferença de TPR só de "
    f"{br(w['equal_opp_gap_60mais_menos_menor30'] * 100, 1)} para {br(s['equal_opp_gap_60mais_menos_menor30'] * 100, 1)} p.p.: o efeito persiste por variáveis correlacionadas "
    f"(renda, dependentes, linhas de crédito). Retirar o atributo sensível não elimina a discriminação indireta."
)
end()
