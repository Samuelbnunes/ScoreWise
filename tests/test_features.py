import numpy as np
import pandas as pd
import pytest

from config import MODEL_EXCLUDED_FEATURES, RAW_FILE
from data import clean_static, deduplicate, load_raw, split
from features import QuantileClipper, add_derived, build_preprocessor, output_feature_names


@pytest.fixture(scope="module")
def sample():
    if not RAW_FILE.exists():  # sem o CSV, os testes que dependem dele são pulados (não falham)
        pytest.skip(f"coloque o cs-training.csv em {RAW_FILE}")
    df = deduplicate(clean_static(load_raw()), verbose=False).sample(2000, random_state=42)
    X, y = df.drop(columns="alvo"), df["alvo"]
    return X, y


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_preprocessor_sem_nan_e_com_flags(sample, kind):
    X, _ = sample
    out = build_preprocessor(kind).fit_transform(X)
    assert not out.isna().any().any()
    assert {"renda_ausente", "atraso_codigo_especial", "total_atrasos"} <= set(out.columns)


def test_flag_renda_ausente_correta(sample):
    X, _ = sample
    out = add_derived(X)
    assert (out["renda_ausente"] == X["renda_mensal"].isna().astype(int)).all()


def test_flag_sem_uso_rotativo_correta(sample):
    # ADR-17: utilização zero tem risco próprio; a flag permite monotonia crescente na utilização positiva
    X, _ = sample
    out = add_derived(X)
    assert (out["sem_uso_rotativo"] == (X["utilizacao_limite"] == 0).astype(int)).all()
    assert out["sem_uso_rotativo"].sum() > 0


def test_percentis_aprendidos_so_no_fit():
    treino = pd.DataFrame({"a": np.arange(100, dtype=float)})
    clipper = QuantileClipper(1, 99).fit(treino)
    novo = pd.DataFrame({"a": [-1000.0, 50.0, 1e9]})
    out = clipper.transform(novo)
    assert out[0, 0] == pytest.approx(np.percentile(treino["a"], 1))
    assert out[2, 0] == pytest.approx(np.percentile(treino["a"], 99))
    # os limites não mudam ao transformar dados novos (sem refit)
    assert clipper.upper_[0] == pytest.approx(98.01)


def test_preprocessor_sem_idade_exclui_a_coluna(sample):
    X, _ = sample
    out = build_preprocessor("tree", drop_cols=("idade",)).fit_transform(X)
    assert "idade" not in out.columns and out.shape[1] == 16 and not out.isna().any().any()


def test_preprocessor_final_sem_idade_e_sem_divida_mensal(sample):
    # v0.2.0 (ADR-18 e C5): idade e divida_mensal fora; a ordem das colunas alinha as restrições monotônicas
    X, _ = sample
    out = build_preprocessor("tree", drop_cols=MODEL_EXCLUDED_FEATURES).fit_transform(X)
    assert "idade" not in out.columns and "divida_mensal" not in out.columns
    assert list(out.columns) == output_feature_names(MODEL_EXCLUDED_FEATURES)
    assert out.shape[1] == 15 and not out.isna().any().any()


def test_reason_codes_nao_exibe_idade_e_pega_o_proximo_fator():
    # Guarda de segurança: a idade não está no modelo v0.2 (ADR-18), mas nunca pode virar motivo se voltar.
    from explain import reason_codes

    names = ["idade", "utilizacao_limite", "atrasos_90", "renda_ausente", "razao_divida"]
    shap_row = [2.0, 1.0, 0.8, 0.7, 0.5]  # idade é o maior fator positivo
    values = [30, 0.9, 1, 0, 0.5]  # renda_ausente = 0 (flag inativa) também não pode aparecer
    r = reason_codes(shap_row, names, top=3, values=values)
    assert r == [
        "Uso elevado do limite de crédito disponível",
        "Histórico de atrasos superiores a 90 dias",
        "Alto comprometimento da renda com dívidas",
    ]


NOMES_V02 = ["utilizacao_limite", "razao_divida", "renda_mensal", "renda_per_capita", "atrasos_30_59", "atrasos_60_89",
             "atrasos_90", "linhas_abertas", "financ_imobiliarios", "dependentes", "total_atrasos", "renda_ausente",
             "atraso_codigo_especial", "teve_atraso_grave", "sem_uso_rotativo"]


def _linha(**kw):
    base = dict(utilizacao_limite=0.85, razao_divida=0.55, renda_mensal=3500, renda_per_capita=1750, atrasos_30_59=0,
                atrasos_60_89=0, atrasos_90=0, linhas_abertas=6, financ_imobiliarios=0, dependentes=1, total_atrasos=0,
                renda_ausente=0, atraso_codigo_especial=0, teve_atraso_grave=0, sem_uso_rotativo=0)
    base.update(kw)
    return [base[n] for n in NOMES_V02]


def _shap(**kw):
    return [kw.get(n, 0.0) for n in NOMES_V02]


def test_reason_codes_agrupa_atrasos_em_um_unico_motivo():
    # ADR-21: 2 atrasos de 30–59 dias geram UM motivo de atraso (antes: "Quantidade elevada..." + "Atrasos 30–59")
    from explain import reason_codes

    r = reason_codes(_shap(total_atrasos=0.6, atrasos_30_59=0.1, utilizacao_limite=0.5), NOMES_V02,
                     values=_linha(atrasos_30_59=2, total_atrasos=2))
    assert r == ["Atrasos recentes entre 30 e 59 dias", "Uso elevado do limite de crédito disponível"]
    grave = reason_codes(_shap(total_atrasos=0.4, atrasos_90=0.3, atrasos_30_59=0.1), NOMES_V02,
                         values=_linha(atrasos_90=1, atrasos_30_59=1, total_atrasos=2, teve_atraso_grave=1))
    assert grave == ["Histórico de atrasos superiores a 90 dias"]
    sem_atraso = reason_codes(_shap(total_atrasos=0.3, utilizacao_limite=0.2), NOMES_V02, values=_linha())
    assert sem_atraso == ["Uso elevado do limite de crédito disponível"]  # sem atraso, o grupo nunca vira motivo


def test_reason_codes_renda_ausente_e_sem_uso_rotativo():
    from explain import reason_codes

    r = reason_codes(_shap(renda_mensal=0.3, renda_ausente=0.2, razao_divida=0.25), NOMES_V02, values=_linha(renda_ausente=1))
    assert r == ["Valor elevado de dívidas mensais", "Renda não informada"]  # sem "Renda baixa"; dívida descrita como valor
    r = reason_codes(_shap(sem_uso_rotativo=0.4, utilizacao_limite=0.1), NOMES_V02,
                     values=_linha(sem_uso_rotativo=1, utilizacao_limite=0))
    assert r == ["Ausência de uso de crédito rotativo no histórico"]  # utilização zero nunca é "uso elevado"
    assert reason_codes(_shap(sem_uso_rotativo=0.4), NOMES_V02, values=_linha()) == []  # flag inativa não é motivo
