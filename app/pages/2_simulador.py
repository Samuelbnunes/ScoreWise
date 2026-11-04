"""Simulador: formulário, decisão, gauge da PD, motivos, waterfall SHAP, parecer e histórico."""
import sys
from datetime import datetime
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
for _p in (_APP, _APP.parent / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from config import PROFILES
from explain import explanation_for
from predict import _model_frame, _validate_frame, load_bundle, predict_one
from ui import charts
from ui.components import begin, br, decision_card, empty_state, end, pct, pd_text, reason_chip, section_header

begin("Simulador")
bundle = load_bundle()
THRESH, BAND = bundle["threshold"], bundle["review_band"]
DECISION_PT = {"APROVAR": "✅ Aprovar", "REVISAR": "🟡 Revisar", "RECUSAR": "❌ Recusar"}

# chave, rótulo, tipo, mín, máx, passo, ajuda em linguagem simples
BLOCKS = {
    "Perfil": "👤",
    "Renda e dívidas": "💰",
    "Histórico de pagamentos": "📅",
}
FIELDS = {
    "Perfil": [
        ("idade", "Idade (anos)", "int", 18, 100, 1, "Idade do solicitante. Entra no modelo, mas não é exibida como motivo da decisão."),
        ("dependentes", "Dependentes", "int", 0, 20, 1, "Número de pessoas dependentes financeiramente do solicitante."),
    ],
    "Renda e dívidas": [
        ("renda_mensal", "Renda mensal (unid. do dataset)", "float", 0.0, 1_000_000.0, 100.0, "Renda mensal, na unidade monetária do dataset (não é R$)."),
        ("razao_divida", "Dívida ÷ renda", "float", 0.0, 100.0, 0.05, "Despesas mensais com dívidas divididas pela renda mensal. 0,5 = metade da renda comprometida."),
        ("utilizacao_limite", "Utilização do limite", "float", 0.0, 10.0, 0.05, "Saldo de cartões e linhas rotativas ÷ limite total. 1,0 = limite todo usado."),
        ("linhas_abertas", "Linhas de crédito abertas", "int", 0, 100, 1, "Empréstimos e linhas de crédito ativos."),
        ("financ_imobiliarios", "Financiamentos imobiliários", "int", 0, 50, 1, "Quantidade de financiamentos imobiliários ou linhas com garantia de imóvel."),
    ],
    "Histórico de pagamentos": [
        ("atrasos_30_59", "Atrasos de 30 a 59 dias", "int", 0, 50, 1, "Vezes em que o solicitante atrasou um pagamento de 30 a 59 dias."),
        ("atrasos_60_89", "Atrasos de 60 a 89 dias", "int", 0, 50, 1, "Vezes em que o solicitante atrasou um pagamento de 60 a 89 dias."),
        ("atrasos_90", "Atrasos de 90 dias ou mais", "int", 0, 50, 1, "Vezes em que o solicitante atrasou 90 dias ou mais. É o sinal mais forte de risco."),
    ],
}
ALL_KEYS = [f[0] for fs in FIELDS.values() for f in fs]


def load_profile(name: str) -> None:
    for k, v in PROFILES[name].items():
        st.session_state[f"f_{k}"] = v
    st.session_state["f_sem_renda"] = False
    st.session_state["result"] = None


def current_inputs() -> dict:
    d = {k: st.session_state[f"f_{k}"] for k in ALL_KEYS}
    if st.session_state["f_sem_renda"]:
        d["renda_mensal"] = None
    return d


def profile_name(d: dict) -> str:
    for n, p in PROFILES.items():
        if all(d.get(k) == v for k, v in p.items()):
            return n
    return "Manual"


if "f_idade" not in st.session_state:
    load_profile("Bom")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("uid", 0)

tc = st.columns([2.4, 1, 1, 1], vertical_alignment="center")
with tc[0]:
    section_header("Simulador de proposta", "Use um perfil ou preencha os dados e clique em Analisar.")
tc[1].button("Bom", key="perfil_bom", on_click=load_profile, args=("Bom",), icon=":material/thumb_up:", width="stretch")
tc[2].button("Limítrofe", key="perfil_limitrofe", on_click=load_profile, args=("Limítrofe",), icon=":material/contrast:", width="stretch")
tc[3].button("Arriscado", key="perfil_arriscado", on_click=load_profile, args=("Arriscado",), icon=":material/warning:", width="stretch")

left, right = st.columns([1.05, 1], gap="large")

with left:
    for block, icon in BLOCKS.items():
        with st.container(border=True):
            st.markdown(f"##### {icon} {block}")
            if block == "Renda e dívidas":
                st.checkbox("Renda não informada", key="f_sem_renda", help="Marque se o solicitante não declarou renda. O modelo trata isso como um sinal próprio.")
            cols = st.columns(2)
            for i, (key, label, kind, lo, hi, step, help_) in enumerate(FIELDS[block]):
                with cols[i % 2]:
                    st.number_input(label, lo, hi, step=step, key=f"f_{key}", help=help_,
                                    disabled=(key == "renda_mensal" and st.session_state["f_sem_renda"]),
                                    format="%d" if kind == "int" else "%.2f")
    analyze = st.button("Analisar proposta", type="primary", key="analisar", icon=":material/bolt:", width="stretch")

if analyze:
    d = current_inputs()
    try:
        with st.spinner("Analisando proposta…"):
            res = predict_one(d)
        st.session_state["result"], st.session_state["inputs"] = res, d
        st.session_state["uid"] += 1
        st.session_state["history"].append({
            "Hora": datetime.now().strftime("%H:%M:%S"), "Perfil": profile_name(d), "PD": pd_text(res.pd),
            "Decisão": DECISION_PT[res.decision], "Principal motivo": res.reasons[0] if res.reasons else "—",
        })
    except ValueError as e:
        st.session_state["result"] = None
        st.error(f"Entrada inválida: {e}")

res = st.session_state.get("result")
with right:
    if res is None:
        empty_state("<b>Nenhuma análise ainda.</b><br>Escolha um perfil ou preencha o formulário e clique em <b>Analisar proposta</b>.")
    else:
        detail = {"APROVAR": "risco abaixo da faixa de revisão", "REVISAR": "caso limítrofe: encaminhar para revisão humana",
                  "RECUSAR": "risco acima da faixa de revisão"}[res.decision]
        decision_card(res.decision, pd_text(res.pd), detail, st.session_state["uid"])
        with st.container(border=True):
            st.plotly_chart(charts.gauge(res.pd, THRESH, BAND), width="stretch", config=charts.CFG)
            st.caption("Verde aprova · amarelo revisa · vermelho recusa · linha escura = corte")
        title = ("Fatores de risco com maior peso (impacto pequeno; a proposta foi aprovada)" if res.decision == "APROVAR"
                 else "Principais motivos")
        st.markdown(f"**{title}**")
        if res.reasons:
            total = sum(res.reason_impacts) or 1.0
            for i, (txt, imp) in enumerate(zip(res.reasons, res.reason_impacts), 1):
                reason_chip(i, txt, imp / total)
            st.caption("Peso relativo de cada motivo.", help="Participação do fator entre os que aumentaram o risco (SHAP). A cor vai de âmbar a vermelho conforme o peso.")
        else:
            st.markdown("Nenhum fator elevou o risco acima da média.")
        if res.decision == "REVISAR":
            st.info("Caso limítrofe, encaminhado para revisão humana (LGPD, art. 20; conferir a norma vigente).")

        d = st.session_state["inputs"]
        linhas = "\n".join(f"- {k}: {'não informada' if v is None else v}" for k, v in d.items())
        motivos = "\n".join(f"{i}. {m}" for i, m in enumerate(res.reasons, 1)) or "Nenhum fator elevou o risco acima da média."
        parecer = (
            f"# Parecer de crédito · ScoreWise\n\n"
            f"- Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n- Modelo: v{res.model_version} ({bundle['best_name']})\n"
            f"- **Decisão: {res.decision}**\n- Probabilidade de inadimplência: {pd_text(res.pd)} "
            f"(corte {pct(THRESH)}; faixa de revisão ±{br(BAND * 100, 0)} p.p.)\n\n"
            f"## Principais motivos\n{motivos}\n\n## Dados informados\n{linhas}\n"
            f"- (renda na unidade monetária do dataset; não é R$)\n\n"
            f"## Avisos\nProtótipo acadêmico; premissas de custo ilustrativas; não usar para decisões reais de crédito. "
            f"Decisões automatizadas podem ser revistas por um humano (LGPD, art. 20; conferir a norma vigente).\n"
        )
        st.download_button("Baixar parecer (.md)", parecer, file_name=f"parecer_scorewise_{st.session_state['uid']}.md",
                           mime="text/markdown", icon=":material/download:", key="baixar_parecer")

if res is not None:
    section_header("Por que esta decisão?", "Contribuição de cada variável para o risco deste caso (SHAP). Barras vermelhas aumentam o risco; verdes diminuem.")
    with st.container(border=True):
        X = _model_frame(_validate_frame(pd.DataFrame([st.session_state["inputs"]])), bundle)
        st.plotly_chart(charts.waterfall(explanation_for(bundle, X)), width="stretch", config=charts.CFG)
        st.caption("Unidade: log-odds do modelo cru (antes da calibração). A idade aparece aqui por transparência, mas não é usada como motivo (ADR-13).")

if st.session_state.get("history"):
    section_header("Histórico da sessão")
    st.dataframe(pd.DataFrame(st.session_state["history"][::-1]), hide_index=True, width="stretch")

end(bundle["model_version"])
