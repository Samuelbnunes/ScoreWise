"""ScoreWise: entrypoint. Só navegação; toda a lógica de modelo vem de src/predict.py (ADR-07)."""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
for p in (APP, APP.parent / "src"):  # `ui/` e os módulos de src/ (o pickle referencia `features`)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(page_title="ScoreWise · Decisões de crédito explicáveis", page_icon="💳", layout="wide")

pages = [
    st.Page("pages/1_inicio.py", title="Início", icon=":material/home:", url_path="inicio", default=True),
    st.Page("pages/2_simulador.py", title="Simulador", icon=":material/calculate:", url_path="simulador"),
    st.Page("pages/3_desempenho.py", title="Desempenho", icon=":material/monitoring:", url_path="desempenho"),
    st.Page("pages/4_explicabilidade.py", title="Explicabilidade", icon=":material/lightbulb:", url_path="explicabilidade"),
    st.Page("pages/5_equidade.py", title="Equidade", icon=":material/balance:", url_path="equidade"),
    st.Page("pages/6_sobre.py", title="Sobre", icon=":material/info:", url_path="sobre"),
]
st.navigation(pages, position="hidden").run()
