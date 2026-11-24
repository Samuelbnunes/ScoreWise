"""Início: hero, 4 KPIs (de reports/metrics) e "como funciona"."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
for _p in (_APP, _APP.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from ui.components import CALIBRATION_PT, begin, br, card_grid, ci_text, end, kpi_card, pct, require_metrics, section_header
from ui.theme import TAGLINE

begin("Início")
m = require_metrics("stats.json", "test_results.json", "cv_results.json", "curves.json")
stats, test, cv = m["stats.json"], m["test_results.json"], m["cv_results.json"]
tres = test["decisao_3_faixas"]  # decisão do produto: aprovar / revisar / recusar (ADR-20)
cal_pt = CALIBRATION_PT.get(m["curves.json"].get("calibration_method", "isotonic"), "calibração")

auc = stats["bootstrap"]["modelos"]["lgbm"]["auc"]
sens = {s["perda_mau_aprovado"]: s for s in stats["sensibilidade_custo"]}
ganho = sens[-5]["ganho_pct_sobre_aprovar_todos"]
lo, hi = sens[-3]["ganho_pct_sobre_aprovar_todos"], sens[-10]["ganho_pct_sobre_aprovar_todos"]
n_total = cv["n_train"] + test["n_test"]

st.markdown(
    f'<div class="sw-hero"><h1>{TAGLINE}</h1>'
    f"<p>O ScoreWise estima a probabilidade de inadimplência de cada proposta, decide entre aprovar, revisar ou recusar "
    f"e explica a decisão com até três fatores que mais pesaram, em português simples.</p></div>",
    unsafe_allow_html=True,
)
st.write("")
if st.button("Simular proposta", type="primary", key="cta", icon=":material/arrow_forward:"):
    st.switch_page("pages/2_simulador.py")

section_header("Resultados no conjunto de teste", f"{br(test['n_test'], 0)} propostas nunca vistas no treino · modelo {test['best_model'].upper()} calibrado")
k = st.columns(4)
with k[0]:
    kpi_card("AUC", br(auc["est"], 3), f"IC 95%: {br(auc['ci95'][0], 3)} a {br(auc['ci95'][1], 3)}",
             "Probabilidade de o modelo ordenar um mau pagador acima de um bom. 0,5 = acaso; 1 = perfeito. IC por bootstrap (1.000 reamostragens).")
with k[1]:
    kpi_card("KS", br(test["models"]["lgbm"]["ks"], 3), "separação entre bons e maus",
             "Maior distância entre as distribuições acumuladas de bons e maus pagadores.")
with k[2]:
    kpi_card("Aprovação automática", pct(tres["aprovacao_automatica"]),
             f"+ {pct(tres['por_faixa']['REVISAR']['pct'])} para revisão humana",
             f"Decisão em 3 faixas: aprova se PD < {pct(tres['faixa'][0])}, revisão humana de {pct(tres['faixa'][0])} a {pct(tres['faixa'][1])} "
             f"e recusa acima. Inadimplência dos aprovados automaticamente: {pct(tres['inadimplencia_aprovados_automaticos'])}; "
             f"da população: {pct(test['test_default_rate'], 2)}.")
with k[3]:
    kpi_card("Ganho vs. aprovar todos", f"+{br(ganho, 1)}%", "sob premissas ilustrativas de custo",
             f"Lucro esperado por proposta em relação a aprovar todos. Depende da perda assumida por mau aprovado: de +{br(lo, 1)}% (−3) a +{br(hi, 1)}% (−10).")

section_header("Como funciona")
card_grid([
    ("1", "Dados", f"{br(n_total, 0)} propostas do dataset público <i>Give Me Some Credit</i>. Limpeza, divisão treino/teste antes de qualquer ajuste e pré-processamento dentro do pipeline, sem vazamento."),
    ("2", "Modelo coerente e calibrado", f"LightGBM com restrições monotônicas (mais atraso nunca reduz o risco; mais renda nunca o aumenta) e {cal_pt}: "
          "uma PD de 10% corresponde a cerca de 10% de maus pagadores. A idade não é usada. O corte é escolhido pelo lucro esperado."),
    ("3", "Decisão explicada", "Aprovar, revisar ou recusar, com até 3 motivos distintos (SHAP). A revisão humana fica na faixa em que ela vale mais "
          "do que a decisão automática."),
])

end()
