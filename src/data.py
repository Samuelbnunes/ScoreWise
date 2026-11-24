"""Carga, limpeza estática e partição. Só entra aqui o que NÃO aprende parâmetros dos dados
(imputação por mediana e clip por percentil ficam no Pipeline, ajustados só no treino)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DELINQUENCY_COLS,
    RAW_COLUMNS,
    RAW_FILE,
    RENAME_PT,
    SEED,
    TARGET,
    TEST_SIZE,
)

TARGET_PT = RENAME_PT[TARGET]  # "alvo"
SPECIAL_DELINQUENCY_CODES = (96, 98)


def file_sha256(path: Path = RAW_FILE) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    return df[RAW_COLUMNS]


def clean_static(df: pd.DataFrame) -> pd.DataFrame:
    """age=0 -> NaN; renda 0 -> NaN; 96/98 nas colunas de atraso -> NaN + flag; renomeia para PT."""
    out = df.copy()
    out.loc[out["age"] == 0, "age"] = np.nan
    # Renda 0 = renda não informada (ADR-19): com renda 0, o DebtRatio do dado bruto é dívida absoluta
    # (mediana 930, 93% acima de 1), como no grupo de renda nula (mediana 1.159, 94% acima de 1).
    out.loc[out["MonthlyIncome"] == 0, "MonthlyIncome"] = np.nan

    special = out[DELINQUENCY_COLS].isin(SPECIAL_DELINQUENCY_CODES)
    out["atraso_codigo_especial"] = special.any(axis=1).astype(int)
    for col in DELINQUENCY_COLS:
        out.loc[special[col], col] = np.nan

    return out.rename(columns=RENAME_PT)


def deduplicate(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Roda ANTES do split: duplicatas entre treino e teste inflariam a métrica."""
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"deduplicate: {n_before - len(out)} linhas removidas ({n_before} -> {len(out)})")
    return out


def split(df: pd.DataFrame):
    """Holdout estratificado 80/20. Retorna X_tr, X_te, y_tr, y_te."""
    X = df.drop(columns=[TARGET_PT])
    y = df[TARGET_PT]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
