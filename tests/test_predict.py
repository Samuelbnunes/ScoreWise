import numpy as np
import pandas as pd
import pytest

from config import MODEL_PATH, PROFILES

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="rode `python src/train.py` antes")

from predict import AVISO_RENDA_ZERO, decide, predict_batch, predict_one  # noqa: E402  (import após o skip)


@pytest.mark.parametrize("perfil,esperado", [("Bom", "APROVAR"), ("Limítrofe", "REVISAR"), ("Arriscado", "RECUSAR")])
def test_perfis_da_demo_caem_na_faixa_esperada(perfil, esperado):
    r = predict_one(PROFILES[perfil])
    assert 0.0 <= r.pd <= 1.0
    assert r.decision == esperado


def test_recusa_tem_ate_3_motivos_distintos_e_pd_valida():
    # C8 / ADR-21: até 3 motivos DISTINTOS (sem completar com fatores repetidos)
    r = predict_one(PROFILES["Arriscado"])
    assert r.decision == "RECUSAR" and 0.0 <= r.pd <= 1.0
    assert 1 <= len(r.reasons) <= 3 and all(r.reasons) and len(set(r.reasons)) == len(r.reasons)


def test_renda_informada_nao_gera_motivo_renda_ausente():
    # regressão: SHAP de uma flag = 0 pode ser positivo; o motivo só vale se a flag estiver ativa
    assert "Renda não informada" not in predict_one(PROFILES["Bom"]).reasons
    sem_renda = predict_one({**PROFILES["Arriscado"], "renda_mensal": None})
    assert 0.0 <= sem_renda.pd <= 1.0


@pytest.mark.parametrize(
    "campo,valor",
    [("idade", 10), ("idade", 150), ("renda_mensal", -1), ("utilizacao_limite", 11), ("atrasos_90", -1),
     ("atrasos_90", 1.5), ("dependentes", None)],
)
def test_input_invalido_levanta_valueerror(campo, valor):
    with pytest.raises(ValueError):
        predict_one({**PROFILES["Bom"], campo: valor})


def test_campo_ausente_levanta_valueerror():
    d = dict(PROFILES["Bom"])
    d.pop("dependentes")
    with pytest.raises(ValueError):
        predict_one(d)


def test_renda_nao_numerica_e_rejeitada():
    # antes, "abc" virava renda não informada em silêncio
    with pytest.raises(ValueError, match="numérico"):
        predict_one({**PROFILES["Bom"], "renda_mensal": "abc"})


def test_renda_zero_equivale_a_nao_informada_e_gera_aviso():
    # ADR-19
    zero = predict_one({**PROFILES["Limítrofe"], "renda_mensal": 0})
    nula = predict_one({**PROFILES["Limítrofe"], "renda_mensal": None})
    assert zero.pd == pytest.approx(nula.pd, abs=1e-12)
    assert zero.avisos == [AVISO_RENDA_ZERO] and nula.avisos == []


def test_idade_e_opcional_e_nao_influi_na_pd():
    # ADR-18: a idade não é variável do modelo
    base = predict_one(PROFILES["Limítrofe"]).pd
    d = dict(PROFILES["Limítrofe"])
    d.pop("idade")
    assert predict_one(d).pd == pytest.approx(base, abs=1e-12)
    for idade in (18, 25, 70, 100):
        assert predict_one({**PROFILES["Limítrofe"], "idade": idade}).pd == pytest.approx(base, abs=1e-12)


@pytest.mark.parametrize("perfil", list(PROFILES))
def test_monotonia_renda_e_atrasos(perfil):
    # ADR-17: mais renda nunca aumenta a PD; mais atraso nunca a diminui
    p = PROFILES[perfil]
    renda = predict_batch(pd.DataFrame([{**p, "renda_mensal": r} for r in range(500, 20001, 500)]))["pd"].to_numpy()
    assert (np.diff(renda) <= 1e-12).all()
    for col in ("atrasos_30_59", "atrasos_60_89", "atrasos_90"):
        pds = predict_batch(pd.DataFrame([{**p, col: k} for k in range(0, 6)]))["pd"].to_numpy()
        assert (np.diff(pds) >= -1e-12).all(), col


def test_decide_respeita_as_tres_faixas():
    # ADR-20: faixa [lo, hi] assimétrica (não é mais corte ± banda)
    lo, hi = 0.207, 0.341
    assert decide(0.10, lo, hi) == "APROVAR"
    assert decide(lo, lo, hi) == "REVISAR"
    assert decide(hi, lo, hi) == "REVISAR"
    assert decide(0.35, lo, hi) == "RECUSAR"
    assert decide(0.26, 0.26, 0.26) == "REVISAR" and decide(0.25, 0.26, 0.26) == "APROVAR"  # faixa vazia: lo = hi = corte


@pytest.mark.parametrize("perfil", list(PROFILES))
def test_nenhum_motivo_exibido_menciona_idade(perfil):
    assert not any("etário" in m for m in predict_one(PROFILES[perfil]).reasons)
