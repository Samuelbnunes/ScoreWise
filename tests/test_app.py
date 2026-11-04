import re

import pytest
from streamlit.testing.v1 import AppTest

from config import MODEL_PATH, ROOT

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="rode `python src/train.py` antes")

APP = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["1_inicio", "2_simulador", "3_desempenho", "4_explicabilidade", "5_equidade", "6_sobre"]


def _open(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    return at.switch_page(f"pages/{page}.py").run()


def _card_html(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown if "sw-decision" in m.value and "<style>" not in m.value)


@pytest.mark.parametrize("page", PAGES)
def test_cada_pagina_carrega_sem_excecao(page):
    at = _open(page)
    assert not at.exception, at.exception
    assert not at.error  # nenhum st.error (ex.: métricas ausentes)


@pytest.mark.parametrize("botao,esperado", [("perfil_bom", "APROVAR"), ("perfil_limitrofe", "REVISAR"), ("perfil_arriscado", "RECUSAR")])
def test_simulador_perfis_geram_a_decisao_esperada(botao, esperado):
    at = _open("2_simulador")
    at.button(key=botao).click().run()
    at.button(key="analisar").click().run()
    assert not at.exception, at.exception
    card = _card_html(at)
    assert esperado in card
    assert "<svg" in card  # decisão nunca só por cor: ícone + texto
    assert len(at.dataframe) >= 1  # histórico da sessão


def test_simulador_pd_nos_extremos_e_motivos():
    at = _open("2_simulador")
    at.button(key="perfil_bom").click().run()
    at.button(key="analisar").click().run()
    assert "&lt; 1%" in _card_html(at) or "< 1%" in _card_html(at)

    at.button(key="perfil_arriscado").click().run()
    at.button(key="analisar").click().run()
    chips = [m.value for m in at.markdown if "sw-chip" in m.value and "<style>" not in m.value]
    assert len(chips) == 3
    assert not any("etário" in c for c in chips)  # ADR-13
    assert re.search(r"[\d,]+%", _card_html(at))


def test_simulador_entrada_invalida_mostra_erro_sem_quebrar():
    at = _open("2_simulador")
    at.number_input(key="f_idade").set_value(18).run()
    at.number_input(key="f_utilizacao_limite").set_value(10.0).run()
    at.button(key="analisar").click().run()
    assert not at.exception


def test_motivos_rotulam_peso_relativo():
    at = _open("2_simulador")
    at.button(key="perfil_arriscado").click().run()
    at.button(key="analisar").click().run()
    chips = [m.value for m in at.markdown if "sw-chip" in m.value and "<style>" not in m.value]
    assert chips and all("peso relativo" in c for c in chips)
    assert not any("var(--sw-accent)" in c for c in chips)  # fatores de risco: nunca no verde de acento


def test_fonte_inter_e_servida_localmente():
    fontes = ROOT / "app" / "static" / "fonts"
    for w in (400, 600, 700):
        f = fontes / f"inter-latin-{w}-normal.woff2"
        assert f.exists() and f.read_bytes()[:4] == b"wOF2"
    assert "enableStaticServing = true" in (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in (ROOT / "app" / "ui" / "theme.py").read_text(encoding="utf-8")  # sem @import externo
