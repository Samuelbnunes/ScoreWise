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
REVIEW_BAND = 0.05  # δ da faixa de revisão humana (seção 3.4)
MODEL_VERSION = "0.1.0"
AGE_BINS = [18, 30, 45, 60, 120]

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
    # Fora da tabela 6.6: necessários para que toda recusa tenha 3 motivos (DoD do E5).
    "atraso_codigo_especial": "Registro de atraso com situação especial no histórico",
    "dependentes": "Número de dependentes associado a maior risco",
    "financ_imobiliarios": "Perfil de financiamentos imobiliários associado a maior risco",
}
