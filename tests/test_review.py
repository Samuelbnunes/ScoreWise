import numpy as np
import pytest

from config import COST_MATRIX, REVIEW_PARAMS
from evaluate import band_or_cut, portfolio_3faixas, review_band, theoretical_threshold


def test_faixa_padrao_por_valor_esperado():
    # ADR-20: com +1/-5/-1/0 e o revisor ilustrativo (custo 0,2; 50% dos maus recusados; 80% dos bons aprovados)
    lo, hi = review_band(COST_MATRIX, REVIEW_PARAMS)
    assert lo == pytest.approx(6 / 29) and hi == pytest.approx(14 / 41)  # 0,2069 e 0,3415
    assert lo < theoretical_threshold() < hi


@pytest.mark.parametrize("custo,sens,esperado", [(0.1, 0.5, (0.172, 0.366)), (0.2, 0.7, (0.154, 0.452)), (0.5, 0.7, (0.231, 0.355)),
                                                  (0.2, 0.3, None), (0.5, 0.5, None)])
def test_sensibilidade_da_faixa(custo, sens, esperado):
    b = review_band(COST_MATRIX, {"custo_revisao": custo, "sensibilidade_analista": sens, "especificidade_analista": 0.8})
    if esperado is None:
        assert b is None  # revisão nunca compensa: decisão em 2 faixas
    else:
        assert b[0] == pytest.approx(esperado[0], abs=1e-3) and b[1] == pytest.approx(esperado[1], abs=1e-3)


def test_valor_da_revisao_dentro_e_fora_da_faixa():
    # dentro da faixa, revisar vale mais que a melhor decisão automática; fora dela, não
    c, p = COST_MATRIX, REVIEW_PARAMS
    s, e, k = p["sensibilidade_analista"], p["especificidade_analista"], p["custo_revisao"]
    lo, hi = review_band(c, p)
    for pd_ in np.linspace(0.01, 0.99, 99):
        v_rev = (1 - pd_) * (e * c["approve_good"] + (1 - e) * c["reject_good"]) + pd_ * (s * c["reject_bad"] + (1 - s) * c["approve_bad"]) - k
        v_auto = max((1 - pd_) * c["approve_good"] + pd_ * c["approve_bad"], (1 - pd_) * c["reject_good"] + pd_ * c["reject_bad"])
        assert (v_rev > v_auto) == (lo < pd_ < hi)


def test_band_or_cut_e_portfolio_3faixas():
    assert band_or_cut(None, 0.26) == (0.26, 0.26)
    y = np.array([0, 0, 1, 0, 1, 1])
    pd_ = np.array([0.05, 0.25, 0.30, 0.10, 0.60, 0.20])
    r = portfolio_3faixas(y, pd_, (0.207, 0.341), 0.26)
    assert sum(v["n"] for v in r["por_faixa"].values()) == len(y)
    # 0,05 / 0,10 / 0,20 < 0,207 -> APROVAR; 0,25 / 0,30 -> REVISAR; 0,60 -> RECUSAR
    assert r["por_faixa"]["APROVAR"]["n"] == 3 and r["por_faixa"]["REVISAR"]["n"] == 2 and r["por_faixa"]["RECUSAR"]["n"] == 1
    assert r["aprovacao_automatica"] == pytest.approx(0.5)
