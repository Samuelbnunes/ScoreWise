from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"

RAW_FILE = DATA_RAW / "cs-training.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

SEED = 42
TARGET = "SeriousDlqin2yrs"
TEST_SIZE = 0.20
CV_FOLDS = 5
# Faixa REVISAR por valor esperado (ADR-20): revisar só onde a análise humana vale mais que decidir automaticamente.
# Revisor ILUSTRATIVO (o dataset não tem dados de revisão humana), nas mesmas unidades de COST_MATRIX:
#   custo_revisao: custo de uma análise; sensibilidade_analista: fração dos maus revisados que o analista recusa;
#   especificidade_analista: fração dos bons revisados que o analista aprova.
# Com os valores abaixo e a matriz +1/-5/-1/0, a faixa é [0,207; 0,341] (o corte teórico 2/7 fica dentro dela).
REVIEW_PARAMS = {"custo_revisao": 0.2, "sensibilidade_analista": 0.5, "especificidade_analista": 0.8}
REVIEW_SENSITIVITY = {"custo_revisao": (0.1, 0.2, 0.5), "sensibilidade_analista": (0.3, 0.5, 0.7), "especificidade_analista": (0.8,)}
MODEL_VERSION = "0.2.0"
AGE_BINS = [18, 30, 45, 60, 120]

# --- Modelo v0.2.0 (ADR-15 a ADR-18). Decisões tomadas na validação cruzada do treino (E2 do PLANO_MELHORIAS.md).
# Fora do modelo: idade (atributo sensível, ADR-18) e divida_mensal (= razão × renda; conflita com a monotonia
# decrescente da renda e não perde AUC ao sair: 0,8597 sem × 0,8595 com).
MODEL_EXCLUDED_FEATURES = ("idade", "divida_mensal")
# Sem class_weight (ADR-15): mesma AUC e probabilidade já próxima da calibrada. Hiperparâmetros do projeto mantidos
# (a busca aleatória ganhou só +0,0021 de AUC, abaixo do mínimo de 0,0025 exigido no plano).
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
)
# Restrições monotônicas (ADR-17): +1 = o risco não diminui quando a variável aumenta; -1 = não aumenta.
# Variáveis ausentes daqui ficam sem restrição (inclusive a flag sem_uso_rotativo, que capta o risco da utilização zero).
MONOTONE = {
    "atrasos_30_59": 1,
    "atrasos_60_89": 1,
    "atrasos_90": 1,
    "total_atrasos": 1,
    "teve_atraso_grave": 1,
    "atraso_codigo_especial": 1,
    "utilizacao_limite": 1,
    "razao_divida": 1,
    "renda_mensal": -1,
    "renda_per_capita": -1,
}
MONOTONE_METHOD = "intermediate"  # AUC CV 0,8597 × 0,8592 do "basic", mesma coerência
# Calibração (ADR-16): "none" | "sigmoid" | "isotonic". Sem calibrador, o ECE OOF foi 0,0034 (> 0,002 do critério A4);
# com Platt, 0,0019. A isotônica foi descartada por gerar PD em degraus.
CALIBRATION_METHOD = "sigmoid"

# Premissas ILUSTRATIVAS (o dataset não tem valor, prazo nem taxa do empréstimo).
COST_MATRIX = {
    "approve_good": +1,   # aprovar bom pagador: juros recebidos
    "approve_bad": -5,    # aprovar mau pagador: perda do principal
    "reject_good": -1,    # recusar bom pagador: custo de oportunidade
    "reject_bad": 0,      # recusar mau pagador: evita a perda
}

KAGGLE_COMPETITION = "GiveMeSomeCredit"

# Campos de entrada da demo/API (nomes curtos em PT; o código 96/98 não é informável pelo usuário).
INPUT_FIELDS = [
    "idade",
    "renda_mensal",
    "razao_divida",
    "utilizacao_limite",
    "linhas_abertas",
    "financ_imobiliarios",
    "atrasos_30_59",
    "atrasos_60_89",
    "atrasos_90",
    "dependentes",
]

# Perfis da demo (T6.5): ajustados para cair em APROVAR / REVISAR / RECUSAR; o modelo não é tocado.
PROFILES = {
    "Bom": dict(idade=62, renda_mensal=9000, razao_divida=0.20, utilizacao_limite=0.10, linhas_abertas=8,
                financ_imobiliarios=1, atrasos_30_59=0, atrasos_60_89=0, atrasos_90=0, dependentes=2),
    "Limítrofe": dict(idade=33, renda_mensal=3500, razao_divida=0.55, utilizacao_limite=0.85, linhas_abertas=6,
                      financ_imobiliarios=0, atrasos_30_59=2, atrasos_60_89=0, atrasos_90=0, dependentes=1),
    "Arriscado": dict(idade=24, renda_mensal=1800, razao_divida=1.10, utilizacao_limite=0.98, linhas_abertas=4,
                      financ_imobiliarios=0, atrasos_30_59=2, atrasos_60_89=1, atrasos_90=2, dependentes=0),
}

# Colunas originais esperadas no CSV (além do índice sem nome).
RAW_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Colunas de atraso, onde 96/98 são códigos especiais e não contagens.
DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Nomes curtos em português (seção 2.2).
RENAME_PT = {
    "SeriousDlqin2yrs": "alvo",
    "RevolvingUtilizationOfUnsecuredLines": "utilizacao_limite",
    "age": "idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "atrasos_30_59",
    "DebtRatio": "razao_divida",
    "MonthlyIncome": "renda_mensal",
    "NumberOfOpenCreditLinesAndLoans": "linhas_abertas",
    "NumberOfTimes90DaysLate": "atrasos_90",
    "NumberRealEstateLoansOrLines": "financ_imobiliarios",
    "NumberOfTime60-89DaysPastDueNotWorse": "atrasos_60_89",
    "NumberOfDependents": "dependentes",
}

FEATURE_LABELS_PT = {
    "utilizacao_limite": "Utilização do limite",
    "idade": "Idade",
    "atrasos_30_59": "Atrasos de 30–59 dias",
    "razao_divida": "Razão dívida/renda",
    "renda_mensal": "Renda mensal",
    "linhas_abertas": "Linhas de crédito abertas",
    "atrasos_90": "Atrasos de 90+ dias",
    "financ_imobiliarios": "Financiamentos imobiliários",
    "atrasos_60_89": "Atrasos de 60–89 dias",
    "dependentes": "Dependentes",
    "total_atrasos": "Total de atrasos",
    "teve_atraso_grave": "Teve atraso grave (90+)",
    "renda_per_capita": "Renda per capita",
    "divida_mensal": "Dívida mensal estimada",
    "renda_ausente": "Renda não informada",
    "atraso_codigo_especial": "Código especial de atraso",
    "sem_uso_rotativo": "Sem uso de crédito rotativo",
}

# Textos dos reason codes (tabela da seção 6.6).
REASON_TEXTS = {
    "atrasos_90": "Histórico de atrasos superiores a 90 dias",
    "atrasos_60_89": "Atrasos recentes entre 60 e 89 dias",
    "atrasos_30_59": "Atrasos recentes entre 30 e 59 dias",
    "total_atrasos": "Quantidade elevada de atrasos no histórico",
    "teve_atraso_grave": "Histórico de atrasos superiores a 90 dias",
    "utilizacao_limite": "Uso elevado do limite de crédito disponível",
    "razao_divida": "Alto comprometimento da renda com dívidas",
    "divida_mensal": "Alto comprometimento da renda com dívidas",
    "renda_mensal": "Renda baixa em relação ao perfil",
    "renda_per_capita": "Renda baixa em relação ao perfil",
    "renda_ausente": "Renda não informada",
    "linhas_abertas": "Número de linhas de crédito abertas",
    # Fora da tabela 6.6: fatores que também podem aumentar o risco (desde a v0.2 não há obrigação de completar 3 motivos).
    "atraso_codigo_especial": "Registro de atraso com situação especial no histórico",
    "dependentes": "Número de dependentes associado a maior risco",
    "financ_imobiliarios": "Perfil de financiamentos imobiliários associado a maior risco",
    "sem_uso_rotativo": "Ausência de uso de crédito rotativo no histórico",  # texto aprovado pelo grupo (E1)
}
# Com renda não informada, razao_divida é a dívida mensal em VALOR (semântica do dataset; ADR-19).
REASON_TEXT_DIVIDA_SEM_RENDA = "Valor elevado de dívidas mensais"

# Motivos agrupados (ADR-21): o SHAP de um grupo é a soma do SHAP das variáveis (aditividade) e vira UM motivo.
# Atrasos: o texto segue o atraso mais grave presente (90+ > 60-89 > 30-59). Renda: um único texto.
REASON_GROUPS = {
    "atrasos": ("atrasos_30_59", "atrasos_60_89", "atrasos_90", "total_atrasos", "teve_atraso_grave"),
    "renda": ("renda_mensal", "renda_per_capita"),
}
