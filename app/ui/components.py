"""Componentes de UI (HTML estático + helpers de formatação pt-BR) e o "chrome" comum das páginas."""
import html
import json
from pathlib import Path

import streamlit as st

from config import METRICS_DIR
from ui.theme import CSS, DECISION_COLOR, DECISION_TINT, LOGO_SVG, TAGLINE, TOKENS

PAGES = [
    ("Início", "pages/1_inicio.py"),
    ("Simulador", "pages/2_simulador.py"),
    ("Desempenho", "pages/3_desempenho.py"),
    ("Explicabilidade", "pages/4_explicabilidade.py"),
    ("Equidade", "pages/5_equidade.py"),
    ("Sobre", "pages/6_sobre.py"),
]
FOOTER_TEXT = "Protótipo acadêmico · premissas de custo ilustrativas · não usar para decisões reais de crédito"


# ---------- formatação pt-BR ----------
def br(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float, nd: int = 1) -> str:
    return br(x * 100, nd) + "%"


def pd_text(p: float) -> str:
    """Evita falsa precisão nos extremos (a isotônica satura em 0 e 1)."""
    if p >= 0.99:
        return "> 99%"
    if p <= 0.01:
        return "< 1%"
    return pct(p)


def ci_text(est: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{br(est, nd)} [{br(lo, nd)}; {br(hi, nd)}]"


# ---------- dados (regra de ouro: tudo vem de reports/metrics/*.json) ----------
def metrics(name: str) -> dict | None:
    p = Path(METRICS_DIR) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def require_metrics(*names: str) -> dict:
    out = {n: metrics(n) for n in names}
    missing = [n for n, v in out.items() if v is None]
    if missing:
        st.error(f"Arquivos de métricas ausentes: {', '.join(missing)}. Rode `python src/train.py` e os scripts de avaliação (veja o README).")
        st.stop()
    return out


# ---------- blocos HTML ----------
def _h(s: str) -> str:
    return html.escape(str(s), quote=True)


def kpi_card(label: str, value: str, sub: str = "", help: str = "") -> None:
    st.markdown(
        f'<div class="sw-kpi" title="{_h(help)}"><div class="sw-kpi-label">{_h(label)}</div>'
        f'<div class="sw-kpi-value">{value}</div><div class="sw-kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="sw-section"><h2>{title}</h2>{sub}</div>', unsafe_allow_html=True)


_ICONS = {  # decisão nunca só por cor: ícone + texto
    "APROVAR": '<path d="M7 12.500l3.200 3.200L17 8.800" fill="none" stroke="#fff" stroke-width="2.600" stroke-linecap="round" stroke-linejoin="round"/>',
    "REVISAR": '<path d="M12 7v6.200M12 16.600v.1" fill="none" stroke="#fff" stroke-width="2.800" stroke-linecap="round"/>',
    "RECUSAR": '<path d="M8 8l8 8M16 8l-8 8" fill="none" stroke="#fff" stroke-width="2.600" stroke-linecap="round"/>',
}
_LABEL = {"APROVAR": "Aprovar", "REVISAR": "Revisão humana", "RECUSAR": "Recusar"}


def decision_card(decision: str, pd_str: str, detail: str, uid: int = 0) -> None:
    c, tint = DECISION_COLOR[decision], DECISION_TINT[decision]
    st.markdown(
        f'<div class="sw-decision" data-uid="{uid}" style="border-color:{c};background:{tint}">'
        f'<svg width="56" height="56" viewBox="0 0 24 24" role="img" aria-label="{_LABEL[decision]}"><circle cx="12" cy="12" r="11" fill="{c}"/>{_ICONS[decision]}</svg>'
        f'<div><div class="sw-decision-label" style="color:{c}">{decision}</div>'
        f'<div class="sw-decision-sub">Probabilidade de inadimplência: <b>{pd_str}</b> · {detail}</div></div></div>',
        unsafe_allow_html=True,
    )


def reason_chip(rank: int, text: str, share: float) -> None:
    st.markdown(
        f'<div class="sw-chip"><span class="sw-chip-rank">{rank}</span><span class="sw-chip-text">{_h(text)}</span>'
        f'<span class="sw-chip-bar" title="Participação no impacto (SHAP)"><div style="width:{max(4, round(share * 100))}%"></div></span>'
        f'<span class="sw-chip-pct">{pct(share, 0)}</span></div>',
        unsafe_allow_html=True,
    )


def card_grid(cards: list[tuple[str, str, str]]) -> None:
    """Cards de mesma altura numa grade responsiva. Cada item: (selo, título, texto HTML)."""
    items = "".join(
        f'<div class="sw-gcard"><div class="sw-step">' + (f'<span class="sw-step-n">{n}</span>' if n else "") + f"<h4>{title}</h4><p>{text}</p></div></div>"
        for n, title, text in cards
    )
    st.markdown(f'<div class="sw-grid">{items}</div>', unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f'<div class="sw-empty">{text}</div>', unsafe_allow_html=True)


def footer(version: str = "") -> None:
    v = f" · modelo v{version}" if version else ""
    st.markdown(f'<div class="sw-footer">{FOOTER_TEXT}{v}</div>', unsafe_allow_html=True)


# ---------- chrome comum ----------
def begin(current: str) -> None:
    """CSS global + cabeçalho com marca e navegação. Chamar no topo de cada página."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    with st.container(key="sw-nav"):
        cols = st.columns([2.6] + [1.15] * len(PAGES), vertical_alignment="center")
        cols[0].markdown(
            f'<div class="sw-brand">{LOGO_SVG}<div><div class="sw-brand-name">ScoreWise</div>'
            f'<div class="sw-brand-tag">{TAGLINE}</div></div></div>',
            unsafe_allow_html=True,
        )
        for col, (label, path) in zip(cols[1:], PAGES):
            try:
                col.page_link(path, label=label)
            except Exception:  # página rodada fora do entrypoint (ex.: testes isolados)
                col.markdown(f"**{label}**")
    idx = [p[0] for p in PAGES].index(current) + 2  # +1 da coluna da marca, +1 por ser 1-based
    st.markdown(
        f"<style>.st-key-sw-nav [data-testid='stColumn']:nth-of-type({idx}) [data-testid='stPageLink'] a"
        f"{{background:#E6F4EF;color:{TOKENS['navy']};box-shadow:inset 0 -3px 0 {TOKENS['accent']};}}</style>",
        unsafe_allow_html=True,
    )


def end(version: str = "") -> None:
    footer(version)
