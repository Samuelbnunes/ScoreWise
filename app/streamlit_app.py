"""Demo ScoreWise. Toda a lógica de modelo vem de src/predict.py (ADR-07)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # o pickle do modelo referencia o módulo `features`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import FIGURES_DIR, METRICS_DIR, MODEL_PATH, PROFILES
from explain import explanation_for, waterfall_figure
from predict import _model_frame, _validate_frame, load_bundle, predict_one

st.set_page_config(page_title="ScoreWise", page_icon="💳", layout="wide")

COLORS = {"APROVAR": "#2a8f4f", "REVISAR": "#d9a400", "RECUSAR": "#c1443c"}
ICONS = {"APROVAR": "✅", "REVISAR": "🟡", "RECUSAR": "❌"}
if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Rode `python src/train.py` antes.")
    st.stop()
bundle = load_bundle()
THRESH, BAND = bundle["threshold"], bundle["review_band"]

# (chave, rótulo, tipo, mín, máx, passo)
FIELDS = [
    ("idade", "Idade", "int", 18, 100, 1),
    ("renda_mensal", "Renda mensal", "float", 0.0, 1_000_000.0, 100.0),
    ("razao_divida", "Razão dívida/renda", "float", 0.0, 100.0, 0.05),
    ("utilizacao_limite", "Utilização do limite (0 a 1+)", "float", 0.0, 10.0, 0.05),
    ("linhas_abertas", "Linhas de crédito abertas", "int", 0, 100, 1),
    ("financ_imobiliarios", "Financiamentos imobiliários", "int", 0, 50, 1),
    ("atrasos_30_59", "Atrasos de 30–59 dias", "int", 0, 50, 1),
    ("atrasos_60_89", "Atrasos de 60–89 dias", "int", 0, 50, 1),
    ("atrasos_90", "Atrasos de 90+ dias", "int", 0, 50, 1),
    ("dependentes", "Dependentes", "int", 0, 20, 1),
]


def load_profile():
    name = st.session_state["perfil"]
    if name in PROFILES:
        for k, v in PROFILES[name].items():
            st.session_state[f"f_{k}"] = v
        st.session_state["f_sem_renda"] = False
        st.session_state["result"] = None


if "f_idade" not in st.session_state:
    st.session_state["perfil"] = "Bom"
    load_profile()
    st.session_state.setdefault("result", None)


def band_figure(pd_value: float):
    fig, ax = plt.subplots(figsize=(7, 1.3))
    lo, hi = THRESH - BAND, THRESH + BAND
    xmax = max(0.6, min(1.0, pd_value * 1.15))
    ax.barh([0], [lo], color=COLORS["APROVAR"], alpha=0.25)
    ax.barh([0], [hi - lo], left=lo, color=COLORS["REVISAR"], alpha=0.35)
    ax.barh([0], [xmax - hi], left=hi, color=COLORS["RECUSAR"], alpha=0.25)
    ax.axvline(pd_value, color="black", lw=3)
    ax.text(pd_value, 0.55, f"PD {pd_value:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.axvline(THRESH, color="gray", ls="--", lw=1)
    ax.text(THRESH, -0.55, f"corte {THRESH:.2f}", ha="center", va="top", fontsize=8, color="gray")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.9, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Probabilidade de inadimplência (calibrada)")
    fig.tight_layout()
    return fig


def show_image(name: str, caption: str = ""):
    p = FIGURES_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figura ausente: {name} (rode `python src/evaluate.py` / `explain.py` / `fairness.py`)")


with st.sidebar:
    st.title("💳 ScoreWise")
    st.selectbox("Perfil pré-definido", ["Bom", "Limítrofe", "Arriscado", "Manual"], key="perfil", on_change=load_profile)
    st.caption(f"Modelo v{bundle['model_version']} · {bundle['best_name']} · corte {THRESH:.3f} ± {BAND:.2f}")

tab_a, tab_d, tab_e, tab_q = st.tabs(["Análise", "Desempenho", "Explicabilidade", "Equidade"])

with tab_a:
    st.subheader("Proposta de crédito")
    cols = st.columns(3)
    for i, (key, label, kind, lo, hi, step) in enumerate(FIELDS):
        with cols[i % 3]:
            if key == "renda_mensal":
                sem = st.checkbox("Renda não informada", key="f_sem_renda")
                st.number_input(label, lo, hi, step=step, key=f"f_{key}", disabled=sem)
            else:
                st.number_input(label, lo, hi, step=step, key=f"f_{key}")

    if st.button("Analisar", type="primary"):
        d = {k: st.session_state[f"f_{k}"] for k, *_ in FIELDS}
        if st.session_state["f_sem_renda"]:
            d["renda_mensal"] = None
        try:
            st.session_state["result"] = predict_one(d)
            st.session_state["inputs"] = d
        except ValueError as e:
            st.session_state["result"] = None
            st.error(f"Entrada inválida: {e}")

    r = st.session_state.get("result")
    if r is not None:
        c = COLORS[r.decision]
        st.markdown(
            f"<div style='border-left:8px solid {c};padding:12px 18px;background:{c}18;border-radius:6px'>"
            f"<span style='font-size:1.8rem;font-weight:700;color:{c}'>{ICONS[r.decision]} {r.decision}</span><br>"
            f"Probabilidade de inadimplência: <b>{r.pd:.1%}</b></div>",
            unsafe_allow_html=True,
        )
        fig = band_figure(r.pd)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        if r.decision == "APROVAR":
            st.markdown("**Fatores de risco com maior peso** (impacto pequeno; a proposta foi aprovada):")
        else:
            st.markdown("**Principais motivos:**")
        for m in r.reasons or ["Nenhum fator elevou o risco acima da média."]:
            st.markdown(f"- {m}")
        if r.decision == "REVISAR":
            st.info("Caso limítrofe: encaminhado para revisão humana (LGPD, art. 20; conferir a norma vigente).")

with tab_d:
    st.subheader("Desempenho no conjunto de teste")
    p = METRICS_DIR / "test_results.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "_Rode `python src/evaluate.py`._")
    c1, c2, c3 = st.columns(3)
    with c1:
        show_image("roc.png")
    with c2:
        show_image("calibration.png")
    with c3:
        show_image("profit_curve.png")
    st.caption("Premissas de custo ilustrativas: aprovar bom +1, aprovar mau −5, recusar bom −1, recusar mau 0.")

with tab_e:
    st.subheader("Explicabilidade (SHAP)")
    left, right = st.columns(2)
    with left:
        st.markdown("**Global**: o que mais pesa no modelo")
        show_image("shap_summary.png")
    with right:
        st.markdown("**Caso atual**: contribuição de cada variável (log-odds do modelo cru)")
        if st.session_state.get("result") is not None:
            X = _model_frame(_validate_frame(pd.DataFrame([st.session_state["inputs"]])), bundle)
            wf = waterfall_figure(explanation_for(bundle, X))
            st.pyplot(wf, width="stretch")
            plt.close(wf)
        else:
            st.info("Clique em **Analisar** na aba Análise.")

with tab_q:
    st.subheader("Equidade por faixa etária")
    pj = METRICS_DIR / "fairness.json"
    if pj.exists():
        t = pd.DataFrame(json.loads(pj.read_text(encoding="utf-8"))["por_faixa"]).set_index("faixa")
        st.dataframe(
            t[["n", "taxa_mau_real", "taxa_aprovacao", "disparate_impact", "tpr_bom_aprovado", "equal_opp_diff_vs_ref"]]
            .style.format({"taxa_mau_real": "{:.1%}", "taxa_aprovacao": "{:.1%}", "disparate_impact": "{:.3f}",
                           "tpr_bom_aprovado": "{:.1%}", "equal_opp_diff_vs_ref": "{:+.1%}"}),
            width="stretch",
        )
    show_image("fairness_age.png")
    pm = METRICS_DIR / "fairness.md"
    if pm.exists():
        st.markdown(pm.read_text(encoding="utf-8").split("**Interpretação.**")[-1])

st.divider()
st.caption("Protótipo acadêmico. Premissas de custo ilustrativas. Não usar para decisões reais de crédito.")
