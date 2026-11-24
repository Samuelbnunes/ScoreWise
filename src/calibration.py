"""Calibradores com a mesma interface (`.predict(prob_crua) -> PD`), para o restante do código não depender
do método escolhido (ADR-16). O método vem de CALIBRATION_METHOD em config.py e é decidido pela validação
cruzada (critério A4 do PLANO_MELHORIAS.md): 'none' -> 'sigmoid' -> 'isotonic', nessa ordem de preferência."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("none", "sigmoid", "isotonic")
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class IdentityCalibrator:
    """Sem calibração: a probabilidade do modelo (treinado sem class_weight) já é a PD."""

    method = "none"

    def fit(self, raw, y):
        return self

    def predict(self, raw):
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


class PlattCalibrator:
    """Platt (sigmoide sobre o logit da probabilidade crua): contínua e monotônica."""

    method = "sigmoid"

    def fit(self, raw, y):
        self.lr_ = LogisticRegression(C=1e6).fit(_logit(raw), np.asarray(y))
        return self

    def predict(self, raw):
        return self.lr_.predict_proba(_logit(raw))[:, 1]


class IsotonicCalibrator:
    """Isotônica (em degraus). Último recurso."""

    method = "isotonic"

    def fit(self, raw, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        return self

    def predict(self, raw):
        return self.iso_.predict(raw)


def fit_calibrator(method: str, raw, y):
    """Ajusta o calibrador sobre as predições OOF cruas (cada uma de um modelo que não viu a linha)."""
    classes = {"none": IdentityCalibrator, "sigmoid": PlattCalibrator, "isotonic": IsotonicCalibrator}
    if method not in classes:
        raise ValueError(f"CALIBRATION_METHOD deve ser um de {METHODS}")
    return classes[method]().fit(raw, y)
