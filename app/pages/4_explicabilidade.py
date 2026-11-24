"""Explicabilidade: importância global SHAP (explain.json), beeswarm e política de motivos (ADR-18, ADR-21)."""
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
rc = ex["reason_codes_teste"]["recusar"]
freq = ex.get("frequencia_motivos_teste", {}).get("recusar", {})
top2 = [FEATURE_LABELS_PT.get(k, k) for k in list(ex["importancia_media_abs_shap"])[:2]]

section_header("O que mais pesa na decisão", f"SHAP sobre uma amostra de {br(ex['shap_amostra'], 0)} propostas do teste (modelo cru, em log-odds)")
a, b = st.columns(2, gap="large")
with a, st.container(border=True):
    st.markdown("**Importância global** (média de |SHAP|)")
    st.plotly_chart(charts.importance(ex["importancia_media_abs_shap"], FEATURE_LABELS_PT), width="stretch", config=charts.CFG)
    st.caption(f"{top2[0]} e {top2[1].lower()} concentram o poder de decisão.")
with b, st.container(border=True):
    st.markdown("**Direção do efeito** (cada ponto é uma proposta)")
    p = FIGURES_DIR / "shap_summary.png"
    if p.exists():
        st.image(str(p), width="stretch")
    st.caption("Vermelho = valor alto da variável. À direita do zero, o risco aumenta.")

section_header("Como os motivos são gerados")
with st.container(border=True):
    st.markdown(
        f"Para cada decisão, o ScoreWise lista até **3 motivos distintos**: os fatores que mais *aumentam* o risco, traduzidos para português simples. "
        f"Variáveis que dizem a mesma coisa são **agrupadas** (todas as de atraso viram um único motivo, com o atraso mais grave; renda e renda per capita "
        f"também), para não repetir o mesmo fator com outras palavras. Das **{br(rc['n'], 0)} recusas** do teste: 3 motivos em {br(rc['com_3_motivos'], 0)}, "
        f"2 motivos em {br(rc['com_2_motivos'], 0)}, 1 motivo em {br(rc['com_1_motivo'], 0)} e nenhum motivo em {br(rc['com_0_motivos'], 0)}.\n\n"
        f"**A idade não é usada pelo modelo** (desde a v0.2), então não aparece nos gráficos nem nos motivos; ela só é usada para auditar a equidade. "
        f"Flags só viram motivo quando ativas (\"Renda não informada\" só aparece para quem não informou renda)."
    )
    if freq:
        st.markdown("**Motivos mais frequentes nas recusas do teste**")
        st.dataframe([{"Motivo": k, "Aparece em": f"{br(v['qualquer_posicao'] / rc['n'] * 100, 1)}%",
                       "É o 1º motivo em": f"{br(v['primeiro_motivo'] / rc['n'] * 100, 1)}%"} for k, v in freq.items()],
                     hide_index=True, width="stretch")
end()
