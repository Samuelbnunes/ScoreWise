import pytest

from config import MODEL_PATH, PROFILES

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="rode `python src/train.py` antes")

from predict import decide, predict_one  # noqa: E402  (import após o skip, que protege o load do modelo)


@pytest.mark.parametrize("perfil,esperado", [("Bom", "APROVAR"), ("Limítrofe", "REVISAR"), ("Arriscado", "RECUSAR")])
def test_perfis_da_demo_caem_na_faixa_esperada(perfil, esperado):
    r = predict_one(PROFILES[perfil])
    assert 0.0 <= r.pd <= 1.0
    assert r.decision == esperado


def test_recusa_tem_3_motivos_e_pd_valida():
    r = predict_one(PROFILES["Arriscado"])
    assert r.decision in {"APROVAR", "REVISAR", "RECUSAR"}
    assert len(r.reasons) == 3 and all(r.reasons)


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
    d.pop("idade")
    with pytest.raises(ValueError):
        predict_one(d)


def test_decide_respeita_as_tres_faixas():
    assert decide(0.10, 0.27, 0.05) == "APROVAR"
    assert decide(0.27, 0.27, 0.05) == "REVISAR"
    assert decide(0.40, 0.27, 0.05) == "RECUSAR"


@pytest.mark.parametrize("perfil", list(PROFILES))
def test_nenhum_motivo_exibido_menciona_idade(perfil):
    assert not any("etário" in m for m in predict_one(PROFILES[perfil]).reasons)
