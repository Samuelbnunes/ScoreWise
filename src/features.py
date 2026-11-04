"""Features derivadas e pré-processamento. Tudo que aprende parâmetros (percentis, medianas,
média/desvio) fica DENTRO do Pipeline, ajustado só nos dados de treino de cada fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DERIVED_NEW = [
    "total_atrasos",
    "teve_atraso_grave",
    "renda_per_capita",
    "divida_mensal",
    "renda_ausente",
]

# Contínuas com cauda absurda: clip p1–p99. Contagens não entram (p99 de atrasos zeraria o sinal).
CLIP_COLS = ["utilizacao_limite", "razao_divida", "renda_mensal", "renda_per_capita", "divida_mensal"]
COUNT_COLS = [
    "idade",
    "atrasos_30_59",
    "atrasos_60_89",
    "atrasos_90",
    "linhas_abertas",
    "financ_imobiliarios",
    "dependentes",
    "total_atrasos",
]
FLAG_COLS = ["renda_ausente", "atraso_codigo_especial", "teve_atraso_grave"]


def add_derived(X: pd.DataFrame) -> pd.DataFrame:
    """Sem estado. A flag renda_ausente é criada ANTES da imputação (que vem depois no Pipeline)."""
    out = X.copy()
    atrasos = ["atrasos_30_59", "atrasos_60_89", "atrasos_90"]
    # NaN (código 96/98) conta como 0 no total; a informação fica na flag atraso_codigo_especial.
    out["total_atrasos"] = out[atrasos].fillna(0).sum(axis=1)
    out["teve_atraso_grave"] = (out["atrasos_90"].fillna(0) > 0).astype(int)
    out["renda_per_capita"] = out["renda_mensal"] / (out["dependentes"] + 1)
    out["divida_mensal"] = out["razao_divida"] * out["renda_mensal"]
    out["renda_ausente"] = out["renda_mensal"].isna().astype(int)
    return out


def _derived_names(_transformer, names_in):
    return np.array(list(names_in) + DERIVED_NEW, dtype=object)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Aprende os percentis p_low/p_high no fit (só treino) e aplica clip; NaN passa intacto."""

    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        if hasattr(X, "columns"):  # guardar nomes antes de virar array (usado no get_feature_names_out)
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.low, axis=0)
        self.upper_ = np.nanpercentile(X, self.high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor(kind: str = "tree", drop_cols: tuple = ()) -> Pipeline:
    """kind='tree': derivadas -> clip -> imputação. kind='linear': idem + StandardScaler.
    drop_cols: colunas originais a excluir do modelo (ablação; ex.: ("idade",))."""
    if kind not in ("tree", "linear"):
        raise ValueError("kind deve ser 'tree' ou 'linear'")
    clip_cols = [c for c in CLIP_COLS if c not in drop_cols]
    count_cols = [c for c in COUNT_COLS if c not in drop_cols]
    flag_cols = [c for c in FLAG_COLS if c not in drop_cols]
    scale = [("scale", StandardScaler())] if kind == "linear" else []

    clip_pipe = Pipeline(
        [("clip", QuantileClipper(1, 99)), ("impute", SimpleImputer(strategy="median"))] + scale
    )
    count_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))] + scale)

    ct = ColumnTransformer(
        [
            ("clip", clip_pipe, clip_cols),
            ("count", count_pipe, count_cols),
            ("flag", "passthrough", flag_cols),
        ],
        verbose_feature_names_out=False,
    )
    derived = FunctionTransformer(add_derived, feature_names_out=_derived_names)
    return Pipeline([("derived", derived), ("columns", ct)]).set_output(transform="pandas")
