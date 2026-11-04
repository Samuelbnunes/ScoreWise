import numpy as np
import pandas as pd
import pytest

from data import clean_static, deduplicate, load_raw, split
from features import QuantileClipper, add_derived, build_preprocessor


@pytest.fixture(scope="module")
def sample():
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


def test_percentis_aprendidos_so_no_fit():
    treino = pd.DataFrame({"a": np.arange(100, dtype=float)})
    clipper = QuantileClipper(1, 99).fit(treino)
    novo = pd.DataFrame({"a": [-1000.0, 50.0, 1e9]})
    out = clipper.transform(novo)
    assert out[0, 0] == pytest.approx(np.percentile(treino["a"], 1))
    assert out[2, 0] == pytest.approx(np.percentile(treino["a"], 99))
    # os limites não mudam ao transformar dados novos (sem refit)
    assert clipper.upper_[0] == pytest.approx(98.01)
