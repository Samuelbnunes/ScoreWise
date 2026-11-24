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
# Podem vir sem valor: renda ausente vira a flag renda_ausente; idade não entra na decisão (ADR-18),
# é recebida só para auditoria de equidade e validada quando informada.
_OPTIONAL = ("renda_mensal", "idade")
AVISO_RENDA_ZERO = "Renda 0 é tratada como renda não informada."


@dataclass
class PredictionResult:
    pd: float
    score: int | None  # 0–1000, entra na F2
    decision: Decision
    reasons: list[str]  # top-3, só fatores que AUMENTAM o risco
    model_version: str
    reason_impacts: list[float] = field(default_factory=list)  # SHAP (log-odds) de cada motivo, p/ a UI
    avisos: list[str] = field(default_factory=list)  # avisos sobre a entrada (ex.: renda 0 tratada como ausente)


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def _validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in INPUT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Campos ausentes: {missing}")
    raw = df[INPUT_FIELDS]
    out = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    # Texto não numérico nunca vira "não informado" em silêncio: só None/NaN/vazio significam ausência.
    vazio = raw.isna() | raw.apply(lambda s: s.astype(str).str.strip().eq(""))
    for c in INPUT_FIELDS:
        if (out[c].isna() & ~vazio[c]).any():
            raise ValueError(f"'{c}' deve ser numérico")
    for c, (lo, hi) in _BOUNDS.items():
        col = out[c]
        if c in _OPTIONAL:
            col = col.dropna()
        elif col.isna().any():
            raise ValueError(f"'{c}' é obrigatório e deve ser numérico")
        if (col < lo).any() or (hi is not None and (col > hi).any()):
            raise ValueError(f"'{c}' fora da faixa plausível [{lo}, {hi if hi is not None else '∞'}]")
    for c in _COUNTS:
        if (out[c] % 1 != 0).any():
            raise ValueError(f"'{c}' deve ser inteiro")
    out.loc[out["renda_mensal"] == 0, "renda_mensal"] = np.nan  # mesma regra do clean_static (ADR-19)
    return out


def validate_input(d: dict) -> dict:
    """Valida um dict de entrada e devolve os valores como floats (renda None ou 0 -> NaN; idade pode ser None)."""
    row = _validate_frame(pd.DataFrame([{k: d.get(k) for k in INPUT_FIELDS}]))
    return row.iloc[0].to_dict()


def _model_frame(clean: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = clean.copy()
    X["atraso_codigo_especial"] = 0  # 96/98 não é informável; só existe no dado histórico
    return X[bundle["feature_names_in"]]


def decide(pd_value: float, lo: float, hi: float) -> Decision:
    """APROVAR se PD < lo; REVISAR se lo <= PD <= hi; RECUSAR se PD > hi (faixa por valor esperado, ADR-20)."""
    if pd_value < lo:
        return "APROVAR"
    if pd_value <= hi:
        return "REVISAR"
    return "RECUSAR"


def review_limits(bundle: dict) -> tuple[float, float]:
    """(lo, hi) da faixa REVISAR do modelo; faixa vazia -> decisão em 2 faixas no corte (lo = hi = corte)."""
    band = bundle.get("review_band")
    return (float(band[0]), float(band[1])) if band else (float(bundle["threshold"]), float(bundle["threshold"]))


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
            "decision": [decide(p, *review_limits(b)) for p in pds],
        },
        index=df.index,
    )
    reasons = [reason_codes_with_impact(r, names, values=v) for r, v in zip(sv, Xt.to_numpy())]
    for i in range(3):
        out[f"reason_{i + 1}"] = [r[i][0] if len(r) > i else "" for r in reasons]
        out[f"impact_{i + 1}"] = [r[i][1] if len(r) > i else 0.0 for r in reasons]
    return out


def _avisos(d: dict) -> list[str]:
    r = d.get("renda_mensal")
    try:
        if r is not None and str(r).strip() != "" and float(r) == 0:
            return [AVISO_RENDA_ZERO]
    except (TypeError, ValueError):
        pass
    return []


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
        avisos=_avisos(d),
    )
