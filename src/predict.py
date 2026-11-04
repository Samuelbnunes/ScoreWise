"""Único ponto de inferência (ADR-07): validação, PD calibrada, decisão em 3 faixas e reason codes."""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from config import INPUT_FIELDS, MODEL_PATH
from explain import reason_codes_with_impact, shap_values_for

Decision = Literal["APROVAR", "REVISAR", "RECUSAR"]

# (mínimo, máximo) plausíveis; None = sem limite. Contagens também devem ser inteiras.
_BOUNDS = {
    "idade": (18, 100),
    "renda_mensal": (0, None),
    "razao_divida": (0, None),
    "utilizacao_limite": (0, 10),
    "linhas_abertas": (0, None),
    "financ_imobiliarios": (0, None),
    "atrasos_30_59": (0, None),
    "atrasos_60_89": (0, None),
    "atrasos_90": (0, None),
    "dependentes": (0, None),
}
_COUNTS = ["linhas_abertas", "financ_imobiliarios", "atrasos_30_59", "atrasos_60_89", "atrasos_90", "dependentes"]


@dataclass
class PredictionResult:
    pd: float
    score: int | None  # 0–1000, entra na F2
    decision: Decision
    reasons: list[str]  # top-3, só fatores que AUMENTAM o risco
    model_version: str
    reason_impacts: list[float] = field(default_factory=list)  # SHAP (log-odds) de cada motivo, p/ a UI


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in INPUT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Campos ausentes: {missing}")
    out = df[INPUT_FIELDS].apply(pd.to_numeric, errors="coerce").astype(float)
    for c, (lo, hi) in _BOUNDS.items():
        col = out[c]
        if c == "renda_mensal":  # renda pode ser desconhecida (NaN): vira a flag renda_ausente
            col = col.dropna()
        elif col.isna().any():
            raise ValueError(f"'{c}' é obrigatório e deve ser numérico")
        if (col < lo).any() or (hi is not None and (col > hi).any()):
            raise ValueError(f"'{c}' fora da faixa plausível [{lo}, {hi if hi is not None else '∞'}]")
    for c in _COUNTS:
        if (out[c] % 1 != 0).any():
            raise ValueError(f"'{c}' deve ser inteiro")
    return out


def validate_input(d: dict) -> dict:
    """Valida um dict de entrada e devolve os valores como floats (renda None -> NaN)."""
    row = _validate_frame(pd.DataFrame([{k: d.get(k) for k in INPUT_FIELDS}]))
    return row.iloc[0].to_dict()


def _model_frame(clean: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = clean.copy()
    X["atraso_codigo_especial"] = 0  # 96/98 não é informável; só existe no dado histórico
    return X[bundle["feature_names_in"]]


def decide(pd_value: float, threshold: float, band: float) -> Decision:
    if pd_value < threshold - band:
        return "APROVAR"
    if pd_value <= threshold + band:
        return "REVISAR"
    return "RECUSAR"


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas: pd, decision, reason_1..3. Mesma lógica do predict_one."""
    b = load_bundle()
    X = _model_frame(_validate_frame(df), b)
    pds = b["calibrator"].predict(b["raw_pipeline"].predict_proba(X)[:, 1])
    sv, Xt = shap_values_for(b, X)
    names = list(b["feature_names_out"])
    out = pd.DataFrame(
        {
            "pd": pds,
            "decision": [decide(p, b["threshold"], b["review_band"]) for p in pds],
        },
        index=df.index,
    )
    reasons = [reason_codes_with_impact(r, names, values=v) for r, v in zip(sv, Xt.to_numpy())]
    for i in range(3):
        out[f"reason_{i + 1}"] = [r[i][0] if len(r) > i else "" for r in reasons]
        out[f"impact_{i + 1}"] = [r[i][1] if len(r) > i else 0.0 for r in reasons]
    return out


def predict_one(d: dict) -> PredictionResult:
    b = load_bundle()
    r = predict_batch(pd.DataFrame([validate_input(d)])).iloc[0]
    return PredictionResult(
        pd=float(r["pd"]),
        score=None,
        decision=r["decision"],
        reasons=[r[f"reason_{i}"] for i in (1, 2, 3) if r[f"reason_{i}"]],
        model_version=b["model_version"],
        reason_impacts=[float(r[f"impact_{i}"]) for i in (1, 2, 3) if r[f"reason_{i}"]],
    )
