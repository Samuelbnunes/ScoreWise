"""Explicabilidade: importância global SHAP (explain.json), beeswarm e política de motivos (ADR-13)."""
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
for _p in (_APP, _APP.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from config import FEATURE_LABELS_PT, FIGURES_DIR
from ui import charts
from ui.components import begin, br, end, section_header, require_metrics

begin("Explicabilidade")
ex = require_metrics("explain.json")["explain.json"]
rc = ex["reason_codes_recusas_teste"]["recusar_pd_ge_corte"]

section_header("O que mais pesa na decisão", f"SHAP sobre uma amostra de {br(ex['shap_amostra'], 0)} propostas do teste (modelo cru, em log-odds)")
a, b = st.columns(2, gap="large")
with a, st.container(border=True):
    st.markdown("**Importância global** (média de |SHAP|)")
    st.plotly_chart(charts.importance(ex["importancia_media_abs_shap"], FEATURE_LABELS_PT), width="stretch", config=charts.CFG)
    st.caption("Utilização do limite e total de atrasos concentram o poder de decisão.")
with b, st.container(border=True):
    st.markdown("**Direção do efeito** (cada ponto é uma proposta)")
    p = FIGURES_DIR / "shap_summary.png"
    if p.exists():
        st.image(str(p), width="stretch")
    st.caption("Vermelho = valor alto da variável. À direita do zero, o risco aumenta.")

section_header("Como os motivos são gerados")
with st.container(border=True):
    st.markdown(
        f"Para cada decisão, o ScoreWise lista até **3 motivos**: os fatores que mais *aumentam* o risco, traduzidos para português simples. "
        f"No teste, **{br(rc['com_3_motivos'], 0)} de {br(rc['n_recusas'], 0)} recusas** têm 3 motivos.\n\n"
        f"**A idade não é exibida como motivo (ADR-13).** Ela continua no modelo e nos gráficos de SHAP, por transparência, mas dizer a alguém que foi "
        f"recusado pelo \"perfil etário\" é problemático ética e juridicamente; entra o próximo fator positivo. Flags só viram motivo quando ativas "
        f"(\"Renda não informada\" só aparece para quem não informou renda)."
    )
end()
