import numpy as np
import pandas as pd
import pytest

from config import DELINQUENCY_COLS, RAW_COLUMNS
from data import clean_static, deduplicate, file_sha256, load_raw, split


def _raw_sample(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {c: rng.integers(0, 3, n) for c in RAW_COLUMNS}, index=range(1, n + 1)
    )
    df["SeriousDlqin2yrs"] = (np.arange(n) % 10 == 0).astype(int)  # 10% de positivos
    df["age"] = rng.integers(21, 80, n)
    return df.astype({"age": float})


def test_load_raw_falha_com_coluna_faltando(tmp_path):
    df = _raw_sample().drop(columns=["DebtRatio"])
    path = tmp_path / "x.csv"
    df.to_csv(path)
    with pytest.raises(ValueError, match="DebtRatio"):
        load_raw(path)


def test_clean_static_converte_codigos_e_idade_zero_em_nan():
    df = _raw_sample()
    df.loc[1, "age"] = 0
    df.loc[2, DELINQUENCY_COLS[0]] = 98
    df.loc[3, DELINQUENCY_COLS[2]] = 96
    out = clean_static(df)
    assert out["idade"].isna().sum() == 1
    assert out["atrasos_30_59"].isna().sum() == 1
    assert out["atrasos_90"].isna().sum() == 1
    assert out["atraso_codigo_especial"].sum() == 2
    assert not out[["atrasos_30_59", "atrasos_60_89", "atrasos_90"]].isin([96, 98]).any().any()


def test_clean_static_trata_renda_zero_como_ausente():
    # ADR-19: renda 0 = renda não informada (no dado bruto, com renda 0 o DebtRatio é dívida em valor)
    df = _raw_sample()
    df.loc[1, "MonthlyIncome"] = 0
    df.loc[2, "MonthlyIncome"] = np.nan
    df.loc[3, "MonthlyIncome"] = 2500
    n_zero_ou_nulo = int(((df["MonthlyIncome"] == 0) | df["MonthlyIncome"].isna()).sum())
    out = clean_static(df)
    assert out["renda_mensal"].isna().sum() == n_zero_ou_nulo
    assert not (out["renda_mensal"] == 0).any()
    assert out.loc[3, "renda_mensal"] == 2500


def test_deduplicate_e_split_estratificado():
    df = clean_static(_raw_sample())
    dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
    out = deduplicate(dup, verbose=False)
    assert len(out) <= len(df)
    X_tr, X_te, y_tr, y_te = split(out)
    assert len(X_tr) + len(X_te) == len(out)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert "alvo" not in X_tr.columns


def test_file_sha256(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
