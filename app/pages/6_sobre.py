"""Sobre: metodologia, premissas de custo, LGPD, limitações, roadmap e créditos."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
for _p in (_APP, _APP.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from ui.components import begin, br, card_grid, end, require_metrics, section_header
from ui.theme import TAGLINE

begin("Sobre")
test = require_metrics("test_results.json")["test_results.json"]
c = test["cost_matrix"]

section_header("Sobre o ScoreWise", f"{TAGLINE}. Projeto acadêmico de modelo de aprovação de crédito com Machine Learning. Protótipo: premissas de custo ilustrativas.")

section_header("Metodologia")
card_grid([
    ("", "Dados", "Dataset público <i>Give Me Some Credit</i> (Kaggle, 2011). Duplicadas removidas e anomalias tratadas (idade 0, códigos 96/98, renda nula) antes de qualquer ajuste."),
    ("", "Sem vazamento", "Divisão 80/20 estratificada antes de tudo. Mediana, clip e escala são aprendidos só no treino, dentro do Pipeline. O teste é usado uma única vez."),
    ("", "Modelo", "LightGBM com <code>class_weight</code>, validação cruzada 5-fold e calibração isotônica sobre as predições out-of-fold. Comparado a um piso (dummy) e à regressão logística."),
    ("", "Decisão", "Corte pelo lucro esperado, com faixa de revisão humana. Motivos via SHAP em português simples; a idade não é exibida como motivo."),
])

section_header("Premissas de custo (ilustrativas)", "O dataset não tem valor, prazo nem taxa do empréstimo; dados reais exigiriam esses campos")
st.dataframe(pd.DataFrame(
    [{"Decisão": "Aprovar", "Real: bom pagador": f"+{c['approve_good']} (juros)", "Real: mau pagador": f"{c['approve_bad']} (perda do principal)"},
     {"Decisão": "Recusar", "Real: bom pagador": f"{c['reject_good']} (oportunidade perdida)", "Real: mau pagador": f"{c['reject_bad']}"}]),
    hide_index=True, width="stretch")
st.caption(f"Com PD calibrada, o corte teórico é 2/7 ≈ {br(2 / 7, 4)}. A renda do dataset está na sua unidade monetária original (não é R$).")

a, b = st.columns(2, gap="large")
with a:
    section_header("LGPD e revisão humana")
    with st.container(border=True):
        st.markdown(
            "O art. 20 da LGPD prevê o direito à revisão de decisões automatizadas e a informações sobre os critérios. No ScoreWise, os motivos em "
            "linguagem simples e a faixa de **revisão humana** em torno do corte respondem *parcialmente* a esse direito, o que não equivale a conformidade jurídica.\n\n"
            "**Confira a norma vigente** antes de citar (LGPD, Lei 13.709/2018; CDC e Lei do Cadastro Positivo)."
        )
with b:
    section_header("Roadmap")
    with st.container(border=True):
        st.markdown(
            "**Fase 2 (rigor):** ablação sem features derivadas, XGBoost e Optuna, scorecard (WoE/IV), segundo dataset (equidade por sexo), *model card*.\n\n"
            "**Fase 3 (produção):** API (FastAPI), contêiner, log de auditoria, MLflow, monitoramento de estabilidade (PSI/CSI), retreino e *champion/challenger*."
        )

section_header("Limitações")
st.dataframe(pd.DataFrame([
    {"ID": "L1", "Limitação": "Sem coluna temporal; partição estratificada", "Consequência": "Não avalia estabilidade no tempo"},
    {"ID": "L2", "Limitação": "Sem propostas recusadas", "Consequência": "Sem reject inference empírico"},
    {"ID": "L3", "Limitação": "Custos assumidos", "Consequência": "O lucro é ilustrativo e sensível à premissa"},
    {"ID": "L4", "Limitação": "Equidade só por idade", "Consequência": "Sem análise de sexo ou região"},
    {"ID": "L5", "Limitação": "Dados de outro país e época (2011)", "Consequência": "Distribuições diferentes do Brasil atual"},
    {"ID": "L6", "Limitação": "Protótipo", "Consequência": "Sem segurança, SLA, auditoria ou versionamento reais"},
]), hide_index=True, width="stretch")

section_header("Créditos e referências")
st.markdown(
    "Ke et al. (2017), *LightGBM* · Lundberg & Lee (2017), *SHAP* · Hardt, Price & Srebro (2016), *Equality of Opportunity* · "
    "Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities* · Kaggle (2011), *Give Me Some Credit* · Brasil, Lei 13.709/2018 (LGPD). "
    "Detalhes em `reports/relatorio.md`."
)
end(test["model_version"])
