# ScoreWise: Arquitetura e Plano de Execução

> **Modelo explicável de aprovação de crédito com Machine Learning**
> Versão do documento: 2.0 · Status: MVP em construção · Seed global: `42`

---

## 0. Como usar este documento

O projeto é dividido em **3 fases**. Só a Fase 1 é obrigatória para a entrega. As outras ficam documentadas para mostrar o caminho de evolução (e para pontuar na banca como "trabalhos futuros planejados").

| Fase | Nome | Objetivo | Prazo | Obrigatória |
|---|---|---|---|---|
| **F1** | MVP apresentável | Modelo treinado, avaliado, explicado e com demo ao vivo | 2 dias (~14 h) | ✅ |
| **F2** | Rigor acadêmico | Testes estatísticos, comparação ampliada, scorecard, segundo dataset | +1 a 2 semanas | ⬜ |
| **F3** | Rumo à produção | API, contêiner, rastreabilidade, monitoramento, CI | +2 a 4 semanas | ⬜ |

**Legenda do backlog (seção 12):**
- `E#` = épico · `T#.#` = task · `#.#.#` = micro task
- ⭐ = *stretch* (fazer só se sobrar tempo dentro da fase)
- Cada task tem **estimativa** e **DoD** (*Definition of Done* = critério objetivo de "pronto")

**Regra de ouro:** nenhum número entra no relatório ou nos slides sem vir de uma execução registrada em `reports/metrics/`.

---

## 1. Visão do produto

### 1.1 Problema
Uma instituição recebe propostas de crédito e precisa decidir **aprovar** ou **recusar**. Erros têm custos assimétricos: aprovar um mau pagador custa muito mais que recusar um bom.

### 1.2 Solução
Para cada proposta, o ScoreWise retorna:
1. **PD** (*Probability of Default* = probabilidade de inadimplência) **calibrada**;
2. **Score** de 0 a 1000 (F2), legível para negócio;
3. **Decisão**: aprovar / revisar manualmente / recusar;
4. **Motivos** (*reason codes*): os 3 fatores que mais pesaram, em português simples.

### 1.3 Perguntas de pesquisa (eixo acadêmico)
| ID | Pergunta | Como responder | Fase |
|---|---|---|---|
| Q1 | Um modelo de *gradient boosting* supera a regressão logística de forma estatisticamente significativa? | AUC no teste + IC 95% por bootstrap pareado | F1 (AUC) / F2 (IC) |
| Q2 | Escolher o corte pelo custo gera mais valor que o corte padrão de 0,5 ou que o corte por F1? | Curva de lucro esperado | F1 |
| Q3 | Quais variáveis concentram o poder de decisão? | SHAP global | F1 |
| Q4 | O modelo trata faixas etárias de forma equitativa? | *Disparate impact* e *equal opportunity* | F1 |
| Q5 | Retirar `age` do modelo reduz a disparidade? A que custo de AUC? | Ablação | F2 |
| Q6 | A decisão é sensível às premissas de custo? | Análise de sensibilidade (−3, −5, −10) | F2 |

### 1.4 Critérios de sucesso (metas, não resultados)
| Critério | Meta | Observação |
|---|---|---|
| AUC no teste (modelo final) | ≥ 0,85 | Referência pública: topo do *leaderboard* da competição ≈ 0,87 (conferir) |
| Calibração | Brier menor que o do modelo não calibrado | Brier = erro quadrático médio da probabilidade |
| Lucro esperado | Corte por custo > corte 0,5 | Sob as premissas da seção 3.3 |
| Explicabilidade | 100% das recusas com 3 motivos | Na demo |
| Reprodutibilidade | `python src/train.py` gera o mesmo AUC em duas execuções | Seed 42 + versões fixadas |

---

## 2. Dados

### 2.1 Fonte
*Give Me Some Credit* (Kaggle, 2011). Arquivo usado: **`cs-training.csv`** (~150 mil linhas, rotulado).
⚠️ O `cs-test.csv` **não tem rótulo**: não serve para avaliação. Todo o teste sai de um *holdout* do `cs-training.csv`.

### 2.2 Dicionário de dados
| Coluna original | Nome curto (PT) | Descrição | Equivalente no mercado BR |
|---|---|---|---|
| `SeriousDlqin2yrs` | `alvo` | 1 = atraso ≥ 90 dias nos 2 anos seguintes | Definição de *default* |
| `RevolvingUtilizationOfUnsecuredLines` | `utilizacao_limite` | Saldo de cartão/linhas ÷ limite total | Utilização de rotativo |
| `age` | `idade` | Idade em anos | Idade |
| `NumberOfTime30-59DaysPastDueNotWorse` | `atrasos_30_59` | Nº de atrasos de 30–59 dias | Histórico de atraso (bureau) |
| `DebtRatio` | `razao_divida` | Despesas mensais com dívida ÷ renda mensal | Comprometimento de renda |
| `MonthlyIncome` | `renda_mensal` | Renda mensal (unidade monetária do dataset) | Renda declarada/comprovada |
| `NumberOfOpenCreditLinesAndLoans` | `linhas_abertas` | Nº de empréstimos e linhas abertas | Relacionamentos ativos (SCR) |
| `NumberOfTimes90DaysLate` | `atrasos_90` | Nº de atrasos ≥ 90 dias | Negativação / restrição |
| `NumberRealEstateLoansOrLines` | `financ_imobiliarios` | Nº de financiamentos imobiliários | Garantia real / patrimônio |
| `NumberOfTime60-89DaysPastDueNotWorse` | `atrasos_60_89` | Nº de atrasos de 60–89 dias | Histórico de atraso |
| `NumberOfDependents` | `dependentes` | Nº de dependentes | Composição familiar |

> **O que NÃO existe no dataset** (e vira limitação): data da proposta, valor solicitado, prazo, taxa, tempo de emprego, consultas ao bureau, sexo e região.

### 2.3 Problemas de qualidade conhecidos (confirmar na EDA)
| Problema | Tratamento | Onde |
|---|---|---|
| Coluna de índice sem nome | `index_col=0` | `data.py` |
| `renda_mensal` nula (19,82% medido) | Mediana + flag `renda_ausente` | Pipeline |
| `dependentes` nulo (2,62% medido) | Mediana (0) | Pipeline |
| `idade = 0` (1 linha medida) | Tratar como nulo → mediana | `data.py` |
| Códigos 96/98 nas 3 colunas de atraso (269 linhas; sempre nas 3 colunas juntas; taxa de mau 54,6% no bruto) | Tratar como nulo → mediana + flag `atraso_codigo_especial` | `data.py` |
| `razao_divida` gigante quando a renda é nula (o campo passa a conter a dívida absoluta) | Flag `renda_ausente` + *clip* p1–p99 | Pipeline |
| `utilizacao_limite` > 1 (até valores absurdos) | *Clip* p1–p99 | Pipeline |
| Linhas duplicadas (609 medidas, 0,41%) | Remover **antes** do split | `data.py` |
| Desbalanceamento (6,70% de positivos após dedup) | `class_weight` + corte por custo | Modelagem |

> Regra: limpezas que **não aprendem nada dos dados** (remover duplicadas, `idade=0`→NaN, códigos 96/98→NaN) podem ser feitas antes do split. Tudo que **aprende parâmetros** (mediana, percentis) fica **dentro do Pipeline**, ajustado só no treino.

### 2.4 Estratégia de partição
```
cs-training.csv (deduplicado)
├── 80% TREINO ──► CV estratificada 5-fold
│                  ├─ seleção de modelo
│                  ├─ calibração (CalibratedClassifierCV, cv=5)
│                  └─ escolha do corte (predições out-of-fold)
└── 20% TESTE ───► usado UMA única vez, para o relatório final
```
- Estratificado por `alvo`, `random_state=42`.
- **Por que não temporal?** O dataset não tem data. Registrado como limitação L1.
- **Por que out-of-fold?** Escolher o corte ou calibrar no próprio teste vaza informação e infla o resultado.

---

## 3. Definição do problema de negócio

### 3.1 Alvo
`alvo = 1` (mau pagador): atraso ≥ 90 dias em até 2 anos após a observação. A janela de performance é fixa (definida pelo dataset). Não há janela de observação explícita.

### 3.2 Unidade de decisão
Uma proposta = uma linha. A decisão é tomada **no momento da proposta**, usando só informações disponíveis nesse instante (todas as colunas do GMSC atendem).

### 3.3 Matriz de custos (premissas ilustrativas)
| | Real: bom (0) | Real: mau (1) |
|---|---|---|
| **Aprovar** | +1 (juros) | −5 (perda do principal) |
| **Recusar** | −1 (oportunidade perdida) | 0 |

**Corte teórico** (com PD bem calibrada): aprovar se o valor esperado de aprovar for maior que o de recusar:

```
(1−p)·(+1) + p·(−5)  >  (1−p)·(−1) + p·0
⇒  2 > 7p  ⇒  p < 2/7 ≈ 0,286
```

O corte empírico (varredura na seção 6.4) deve ficar próximo de 0,286. Se não ficar, é sinal de **calibração ruim**. Esse cruzamento entre teoria e prática é um bom argumento na apresentação.

### 3.4 Decisão em três faixas (suporte à LGPD, art. 20)
| PD | Decisão |
|---|---|
| `PD < corte − δ` | ✅ Aprovar |
| `corte − δ ≤ PD ≤ corte + δ` | 🟡 Revisão humana |
| `PD > corte + δ` | ❌ Recusar |

δ padrão = 0,05 (configurável). A revisão humana em casos limítrofes é a resposta prática ao direito de revisão de decisões automatizadas.

---

## 4. Decisões de arquitetura (ADRs resumidos)

| ID | Decisão | Alternativa descartada | Motivo | Fase |
|---|---|---|---|---|
| ADR-01 | Split estratificado 80/20 + CV 5-fold | Split temporal | Sem coluna de data | F1 |
| ADR-02 | LightGBM como modelo principal | XGBoost/CatBoost | Rápido, nativo com NaN, bom em tabular; os outros vão para a F2 | F1 |
| ADR-03 | Desbalanceamento via `class_weight` + corte por custo | SMOTE | SMOTE distorce probabilidades e cria vazamento se feito fora do Pipeline | F1 |
| ADR-04 | Calibração isotônica com `CalibratedClassifierCV(cv=5)` | Platt | ~120 mil linhas é volume suficiente para a isotônica; Platt fica como comparação | F1 |
| ADR-05 | SHAP sobre o modelo **não calibrado** | SHAP no calibrado | O `TreeExplainer` exige árvore pura; a calibração é monotônica, então não muda a ordem dos fatores | F1 |
| ADR-06 | Artefato único `model.joblib` com metadados | Arquivos soltos | Contrato único para demo, API e testes | F1 |
| ADR-07 | `src/predict.py` como **único ponto de inferência** | Lógica duplicada no app | Demo e futura API usam o mesmo código | F1 |
| ADR-08 | Demo em Streamlit | FastAPI + front | Velocidade e visual para a banca | F1 |
| ADR-09 | Equidade medida só por faixa etária | Sexo/região | Atributos inexistentes no GMSC; o UCI cobre sexo na F2 | F1 |
| ADR-10 | Scripts em `src/` (sem pacote) | Pacote instalável | Simplicidade no MVP; vira pacote `scorewise` na F3 | F1→F3 |
| ADR-11 | MLflow para rastrear experimentos | Planilha | Padrão de mercado, roda local | F3 |
| ADR-12 | FastAPI + Pydantic para servir | Flask | Validação de contrato e OpenAPI automáticos | F3 |

---

## 5. Arquitetura lógica

### 5.1 Fluxo de treino (offline)
```
data/raw/cs-training.csv
        │  data.py: load_raw → clean_static → dedup → split
        ▼
  X_train, y_train ───────────────────────────┐        X_test, y_test
        │                                     │              │
        ▼                                     │              │
  features.py: build_preprocessor()           │              │
  (derivadas + imputação + clip + flags)      │              │
        │                                     │              │
        ▼                                     │              │
  train.py: Pipeline(pre + modelo)            │              │
   ├─ CV 5-fold → métricas por fold           │              │
   ├─ OOF probs → calibração → corte por custo│              │
   └─ fit final no treino inteiro ◄───────────┘              │
        │                                                    │
        ▼                                                    ▼
  models/model.joblib  ──────────────────────►  evaluate.py (teste, UMA vez)
                                                 ├─ reports/metrics/*.json
                                                 └─ reports/figures/*.png
                                                       │
                                     explain.py / fairness.py
                                     (SHAP, reason codes, equidade)
```

### 5.2 Fluxo de inferência (online / demo)
```
Formulário Streamlit ─► predict.py::predict_one(dict)
   (F3: POST /v1/predict)        ├─ valida campos (faixas plausíveis)
                                 ├─ DataFrame com o schema do treino
                                 ├─ PD calibrada
                                 ├─ decisão em 3 faixas
                                 ├─ top-3 reason codes (SHAP local)
                                 └─ retorna PredictionResult
```

### 5.3 Contrato do artefato `models/model.joblib`
```python
{
    "calibrated_model": CalibratedClassifierCV,  # usado para a PD
    "raw_pipeline": Pipeline,                    # usado para o SHAP
    "threshold": float,                          # corte por custo (OOF)
    "review_band": 0.05,                         # δ da revisão humana
    "feature_names_in": list[str],               # schema de entrada
    "feature_names_out": list[str],              # após o preprocessamento
    "metrics_cv": dict,                          # médias e desvios da CV
    "cost_matrix": dict,
    "model_version": "0.1.0",
    "trained_at": "ISO-8601",
    "data_sha256": str,                          # hash do CSV de origem
    "seed": 42,
    "lib_versions": {"sklearn": ..., "lightgbm": ..., "shap": ...},
}
```

### 5.4 Contrato de saída da predição
```python
@dataclass
class PredictionResult:
    pd: float                 # 0–1, calibrada
    score: int | None         # 0–1000 (F2)
    decision: Literal["APROVAR", "REVISAR", "RECUSAR"]
    reasons: list[str]        # top-3, em PT, só fatores que AUMENTAM o risco
    model_version: str
```

---

## 6. Especificação dos componentes

### 6.1 `src/config.py` ✅ (já existe)
Caminhos (`pathlib`), `SEED`, `TARGET`, `TEST_SIZE`, `COST_MATRIX`. **Adicionar:** `CV_FOLDS=5`, `REVIEW_BAND=0.05`, `MODEL_VERSION`, `AGE_BINS=[18,30,45,60,120]`, `METRICS_DIR`, mapa `FEATURE_LABELS_PT` e mapa `REASON_TEXTS`.

### 6.2 `src/data.py`
| Função | Entrada → Saída | Regras |
|---|---|---|
| `load_raw(path)` | CSV → `DataFrame` | `index_col=0`; valida as 11 colunas esperadas (falha alto se faltar alguma) |
| `clean_static(df)` | df → df | `age==0`→NaN; 96/98→NaN nas colunas de atraso + flag; renomeia para PT (opcional) |
| `deduplicate(df)` | df → df | Loga quantas linhas saíram |
| `split(df)` | df → `X_tr, X_te, y_tr, y_te` | Estratificado, seed 42 |
| `file_sha256(path)` | path → str | Rastreabilidade |

### 6.3 `src/features.py`
- `add_derived(X)` (dentro de um `FunctionTransformer`, sem estado):
  - `total_atrasos = 30_59 + 60_89 + 90`
  - `teve_atraso_grave = atrasos_90 > 0`
  - `renda_per_capita = renda / (dependentes + 1)`
  - `divida_mensal = razao_divida × renda`
  - `renda_ausente = renda.isna()` (criada **antes** da imputação)
- `build_preprocessor()` → `Pipeline([derived, ColumnTransformer(imputer mediana + clip p1–p99 → numéricas; passthrough → flags)])`
- Clip p1–p99 implementado como transformador próprio (`QuantileClipper`, com `fit` que aprende os percentis no treino).
- Para a logística: acrescentar `StandardScaler` (as árvores não precisam).

### 6.4 `src/train.py`
1. Carrega, limpa e divide (via `data.py`).
2. Define os candidatos:
   - `dummy`: `DummyClassifier(strategy="prior")` (piso de referência)
   - `logreg`: `LogisticRegression(class_weight="balanced", max_iter=1000)`
   - `lgbm`: `LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1)`
3. CV 5-fold estratificada → AUC, Gini, KS, PR-AUC, Brier por fold (média ± desvio).
4. Escolhe o melhor por AUC médio na CV.
5. `CalibratedClassifierCV(best, method="isotonic", cv=5)` → OOF probs via `cross_val_predict`.
6. **Corte por custo:** varre `t ∈ [0,01; 0,99]` (passo 0,005) sobre as OOF calibradas e escolhe `argmax(lucro)`. Registra também o corte teórico 0,286.
7. Ajusta o modelo final no treino inteiro e salva o artefato (5.3) + `reports/metrics/cv_results.json`.

> ⚠️ `class_weight="balanced"` desloca as probabilidades para cima. A calibração **corrige isso**; é por esse motivo que ela é obrigatória aqui, e não opcional.

### 6.5 `src/evaluate.py`
Roda **uma vez** sobre o teste e gera:
| Saída | Arquivo |
|---|---|
| Tabela comparativa (dummy, logreg, lgbm calibrado) | `reports/metrics/test_results.json` + `.md` |
| Curva ROC (todos os modelos) | `figures/roc.png` |
| Curva PR | `figures/pr.png` |
| Curva KS (distribuições acumuladas de bons × maus) | `figures/ks.png` |
| Curva de calibração (antes × depois) | `figures/calibration.png` |
| Curva de lucro × corte, com marcações 0,5 / F1 / custo / 0,286 | `figures/profit_curve.png` |
| Matriz de confusão no corte escolhido | `figures/confusion.png` |
| Taxa de aprovação e inadimplência na carteira aprovada | no `.json` |

**Definições:**
- **AUC**: probabilidade de o modelo ordenar um mau acima de um bom.
- **Gini** = 2·AUC − 1.
- **KS**: maior distância entre as distribuições acumuladas de score de bons e maus.
- **Brier**: média de (p − y)²; quanto menor, melhor a calibração.
- **Inadimplência da carteira aprovada**: % de maus entre os aprovados (métrica que o negócio entende).

### 6.6 `src/explain.py`
- `TreeExplainer(raw_pipeline[-1])` sobre os dados transformados de uma amostra do teste (n=2.000, por desempenho).
- `figures/shap_summary.png` (beeswarm) e `figures/shap_bar.png`.
- `reason_codes(shap_row, top=3)`: pega só as contribuições **positivas** (que aumentam o risco), ordena e traduz via `REASON_TEXTS`:

| Feature | Texto do motivo |
|---|---|
| `atrasos_90` | "Histórico de atrasos superiores a 90 dias" |
| `atrasos_60_89` | "Atrasos recentes entre 60 e 89 dias" |
| `atrasos_30_59` | "Atrasos recentes entre 30 e 59 dias" |
| `total_atrasos` | "Quantidade elevada de atrasos no histórico" |
| `utilizacao_limite` | "Uso elevado do limite de crédito disponível" |
| `razao_divida` | "Alto comprometimento da renda com dívidas" |
| `renda_mensal` / `renda_per_capita` | "Renda baixa em relação ao perfil" |
| `renda_ausente` | "Renda não informada" |
| `idade` | "Perfil etário associado a maior risco" ⚠️ ver 7.3 |
| `linhas_abertas` | "Número de linhas de crédito abertas" |

- ⭐ `figures/shap_waterfall_<perfil>.png` para os 3 perfis da demo.

### 6.7 `src/fairness.py`
- Faixas etárias: `<30`, `30–44`, `45–59`, `60+`.
- Por faixa: n, taxa real de mau, taxa de aprovação, TPR (*recall* dos bons aprovados), FPR.
- **Disparate impact (DI)** = taxa de aprovação do grupo ÷ taxa do grupo de referência (o de maior aprovação). Alerta se DI < 0,8 (regra dos 4/5).
- **Equal opportunity difference** = diferença de TPR entre grupos, considerando como "positivo" o **bom pagador aprovado**.
- Saídas: `reports/metrics/fairness.json` e `figures/fairness_age.png`.
- O texto deve diferenciar **disparidade justificada pelo risco real** (taxa de mau diferente entre faixas) de **disparidade além do risco** (comparar TPR, não só aprovação).

### 6.8 `src/predict.py`
- `load_bundle()` com cache.
- `validate_input(d)`: tipos e faixas plausíveis (idade 18–100, renda ≥ 0, contagens ≥ 0, utilização 0–10).
- `predict_one(d) -> PredictionResult` e `predict_batch(df)`.
- Usado pelo Streamlit (F1) e pela API (F3). **Nenhuma lógica de modelo fora daqui.**

### 6.9 `app/streamlit_app.py`
| Área | Conteúdo |
|---|---|
| Sidebar | Seletor de perfil pré-definido (Bom / Limítrofe / Arriscado / Manual) + versão do modelo |
| Aba **Análise** | Formulário → botão **Analisar** → card com decisão (cor), PD (barra com o corte marcado), score (F2) e 3 motivos |
| Aba **Desempenho** | Tabela comparativa + ROC, calibração e curva de lucro (lidas de `reports/`) |
| Aba **Explicabilidade** | SHAP summary + waterfall do caso atual |
| Aba **Equidade** | Tabela e gráfico por faixa etária |
| Rodapé | "Protótipo acadêmico. Premissas de custo ilustrativas." |

**Perfis pré-definidos** (valores iniciais, ajustar após o treino para que cada um caia na faixa esperada):
| Campo | Bom | Limítrofe | Arriscado |
|---|---|---|---|
| idade | 48 | 33 | 24 |
| renda_mensal | 9.000 | 3.500 | 1.800 |
| razao_divida | 0,20 | 0,55 | 1,10 |
| utilizacao_limite | 0,10 | 0,65 | 0,98 |
| linhas_abertas | 8 | 6 | 4 |
| financ_imobiliarios | 1 | 0 | 0 |
| atrasos 30–59 / 60–89 / 90+ | 0 / 0 / 0 | 1 / 0 / 0 | 2 / 1 / 2 |
| dependentes | 2 | 1 | 0 |

> Micro task obrigatória: rodar os 3 perfis depois do treino e confirmar que caem em APROVAR / REVISAR / RECUSAR. Se não caírem, ajustar os valores (e não o modelo).

### 6.10 Testes (`tests/`)
| Arquivo | Teste | Fase |
|---|---|---|
| `test_data.py` | `load_raw` falha com uma coluna faltando; `clean_static` converte 96/98 e `age=0` em NaN | F1 |
| `test_features.py` | Saída do preprocessor sem NaN; flag `renda_ausente` correta; sem vazamento (percentis aprendidos só no fit) | F1 |
| `test_predict.py` | PD ∈ [0,1]; decisão ∈ {3 valores}; 3 motivos para RECUSAR; input inválido → `ValueError` | F1 |
| `test_api.py` | `/health` 200; `/v1/predict` com payload válido e inválido | F3 |

Configuração: `pytest.ini` com `pythonpath = src` (resolve o `from config import ...`). Fixture com amostra de 2.000 linhas para rodar em segundos.

---

## 7. Governança, ética e regulação

> Tudo nesta seção depende de norma vigente: **conferir antes de citar no relatório.**

### 7.1 Referências normativas
| Norma | Relevância |
|---|---|
| LGPD (Lei 13.709/2018), art. 7º, X | Base legal para tratamento de dados na proteção ao crédito |
| LGPD, art. 20 | Direito à revisão de decisões automatizadas e a informações sobre os critérios |
| LGPD, art. 6º, III | Princípio da necessidade (minimização de dados) |
| Lei do Cadastro Positivo (12.414/2011) | Uso de histórico de pagamentos |
| CDC (Lei 8.078/1990), art. 43 | Bancos de dados de consumidores |
| Resolução CMN 4.557/2017 | Estrutura de gerenciamento de riscos (inclui risco de modelo) |

### 7.2 Como o ScoreWise responde
| Requisito | Implementação |
|---|---|
| Explicação da decisão | Reason codes em linguagem simples |
| Revisão humana | Faixa REVISAR (seção 3.4) |
| Minimização | Só 10 variáveis, sem identificadores pessoais |
| Rastreabilidade | Versão do modelo, hash dos dados e seed no artefato; log de auditoria na F3 |
| Documentação | Model card (F2) |

### 7.3 Pontos éticos para discutir
- **Idade como variável:** é preditiva, mas é um atributo sensível. A pergunta Q5 (ablação) quantifica o trade-off entre desempenho e equidade.
- **Variáveis proxy:** renda e número de dependentes podem carregar informação socioeconômica correlacionada a grupos protegidos.
- **Reject inference:** o modelo só vê quem foi aprovado no passado (viés de seleção). Técnicas a apresentar: *reweighting/augmentation* e *parceling* (sem implementação; o dataset não tem recusados).
- **Feedback loop:** em produção, quem é recusado nunca gera rótulo, o que reforça o viés com o tempo.

---

## 8. Caminho para produção (F3)

### 8.1 Requisitos não funcionais (alvo)
| Requisito | Meta |
|---|---|
| Latência p95 de `/v1/predict` | < 200 ms (CPU) |
| Disponibilidade | Health check + reinício automático pelo contêiner |
| Reprodutibilidade | Imagem Docker com versões fixadas; modelo versionado |
| Auditoria | 100% das decisões logadas |
| Privacidade | Nenhum dado pessoal bruto nos logs (só hash do payload) |

### 8.2 API (FastAPI)
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status + versão do modelo carregado |
| GET | `/v1/model-info` | Métricas, corte, data de treino |
| POST | `/v1/predict` | Uma proposta → `PredictionResult` |
| POST | `/v1/predict/batch` | Até 1.000 propostas |

Exemplo de resposta:
```json
{
  "request_id": "uuid",
  "pd": 0.31,
  "score": 512,
  "decision": "REVISAR",
  "reasons": ["Uso elevado do limite de crédito disponível", "..."],
  "model_version": "0.1.0"
}
```

### 8.3 Log de auditoria (JSON por linha)
`timestamp`, `request_id`, `model_version`, `payload_sha256`, `pd`, `decision`, `threshold`, `reasons`, `latency_ms`.

### 8.4 Rastreamento e versionamento
- **MLflow** (local): cada execução de `train.py` registra parâmetros, métricas e o artefato; registro de modelos com estágios *Staging*/*Production*.
- **DVC** ⭐: versiona `data/raw` por hash.

### 8.5 Monitoramento
- **PSI** (*Population Stability Index*) por score e **CSI** por variável:
  `PSI = Σ (atual% − base%) · ln(atual% / base%)` em 10 faixas.
  < 0,10 estável · 0,10–0,25 atenção · > 0,25 investigar/retreinar.
- Taxa de aprovação diária e PD média (alerta se desviar mais de 3σ).
- Desempenho real (AUC/KS) assim que os rótulos maturarem (2 anos neste dataset, o que mostra por que as métricas de *drift* são o alerta antecipado).
- **Retreino:** trimestral ou disparado por PSI > 0,25.
- **Champion/challenger:** o desafiante recebe 10% do tráfego (ou roda em *shadow mode*) até comprovar superioridade.

### 8.6 CI/CD (GitHub Actions)
`ruff` (lint) → `pytest` → build da imagem Docker → (manual) publicação.

---

## 9. Estrutura de pastas (alvo)

```
ScoreWise/
├── .docs/
│   ├── architecture.md          # este documento
│   └── model_card.md            # F2
├── data/
│   ├── raw/                     # cs-training.csv (fora do git)
│   └── processed/               # opcional: splits em parquet
├── notebooks/
│   ├── 01_eda.ipynb             # F1
│   └── 02_experimentos.ipynb    # F2
├── src/
│   ├── config.py                ✅
│   ├── download_data.py         ✅
│   ├── data.py                  # F1
│   ├── features.py              # F1
│   ├── train.py                 # F1
│   ├── evaluate.py              # F1
│   ├── explain.py               # F1
│   ├── fairness.py              # F1
│   ├── predict.py               # F1 (ponto único de inferência)
│   ├── scorecard.py             # F2
│   ├── stats.py                 # F2 (bootstrap, ablação)
│   └── monitoring.py            # F3 (PSI/CSI)
├── app/
│   └── streamlit_app.py         # F1
├── api/                         # F3
│   ├── main.py
│   └── schemas.py
├── models/
│   └── model.joblib             # fora do git (gerado)
├── reports/
│   ├── figures/
│   ├── metrics/                 # JSONs gerados pelos scripts
│   ├── relatorio.md
│   └── slides/
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_api.py              # F3
├── pytest.ini
├── requirements.txt             ✅
├── README.md
├── .gitignore                   ✅ (adicionar models/*.joblib, mlruns/)
├── Dockerfile                   # F3
├── docker-compose.yml           # F3
└── .github/workflows/ci.yml     # F3
```

---

## 10. Stack

| Camada | F1 | F2 | F3 |
|---|---|---|---|
| Linguagem | Python 3.13 (o mesmo da `.venv`) | | |
| Dados | pandas, numpy, pyarrow | | DVC ⭐ |
| ML | scikit-learn, lightgbm | xgboost, optuna, optbinning | |
| Explicabilidade | shap | | |
| Visualização | matplotlib, seaborn | | |
| Demo | streamlit | | |
| Serviço | | | fastapi, uvicorn, pydantic |
| MLOps | joblib | | mlflow, docker, GitHub Actions |
| Qualidade | pytest | | ruff |

> Antes de cada fase, rodar `pip freeze > requirements.txt` com a `.venv` ativa. Para a F3, separar `requirements-dev.txt`.

---

## 11. Riscos do projeto

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Download pelo Kaggle falhar (token ou regras) | Média | Alto | Download manual do CSV (já previsto no `download_data.py`) |
| Incompatibilidade de versões (Python 3.13 + shap/lightgbm) | Média | Alto | Testar `import shap, lightgbm` na T0.3; se falhar, recriar a `.venv` com 3.11/3.12 |
| SHAP lento | Baixa | Médio | Amostra de 2.000 linhas; pré-calcular as figuras |
| Demo falhar ao vivo | Média | Alto | Perfis pré-definidos + figuras pré-geradas + gravação de vídeo de backup |
| Faltar tempo | Alta | Alto | Corte rígido por ⭐; F2/F3 viram slides de "trabalhos futuros" |
| Perfis não caírem na faixa esperada | Média | Baixo | Micro task de ajuste após o treino |

---

## 12. Backlog: épicos, tasks e micro tasks

### FASE 1: MVP (2 dias, ~14 h)

#### E0. Setup do ambiente · 1 h
- [x] **T0.1** Estrutura de pastas e `.venv`
- [x] **T0.2** `requirements.txt` com versões fixadas
- [x] **T0.3** Validar o ambiente · 15 min
  - [x] 0.3.1 `python -c "import sklearn, lightgbm, shap, streamlit; print('ok')"`
  - [x] 0.3.2 Se `shap` ou `lightgbm` falhar no 3.13 → recriar a `.venv` com 3.12
- [x] **T0.4** `config.py` base
- [x] **T0.5** Completar o `config.py` (itens da seção 6.1) · 15 min
- [x] **T0.6** Obter o dataset · 15 min
  - [x] 0.6.1 Configurar `~/.kaggle/kaggle.json` e aceitar as regras da competição
  - [x] 0.6.2 `python src/download_data.py` (ou download manual para `data/raw/`)
- [x] **T0.7** `pytest.ini` (`pythonpath = src`) + criar `reports/metrics/` · 5 min
- [x] **T0.8** Atualizar o `.gitignore` (`models/*.joblib`, `reports/metrics/*.json` opcional, `mlruns/`) + `git init` + primeiro commit · 10 min
- **DoD:** imports OK, CSV presente, `pytest` roda (mesmo sem testes).

#### E1. Dados e EDA · 2 h 30
- [x] **T1.1** `data.py` · 45 min
  - [x] 1.1.1 `load_raw` com `index_col=0` e validação das colunas
  - [x] 1.1.2 `clean_static`: `age==0`→NaN; 96/98→NaN + flag
  - [x] 1.1.3 `deduplicate` com log da contagem
  - [x] 1.1.4 `split` estratificado (seed 42)
  - [x] 1.1.5 `file_sha256`
- [x] **T1.2** Notebook `01_eda.ipynb` · 1 h 30
  - [x] 1.2.1 Shape, tipos, `describe()`
  - [x] 1.2.2 Tabela de nulos (% por coluna)
  - [x] 1.2.3 Gráfico: distribuição do alvo
  - [x] 1.2.4 Gráfico: taxa de mau por faixa de idade
  - [x] 1.2.5 Gráfico: taxa de mau por nº de atrasos 90+
  - [x] 1.2.6 Gráfico: taxa de mau por decil de utilização do limite
  - [x] 1.2.7 Heatmap de correlação (Spearman)
  - [x] 1.2.8 Evidências das anomalias (96/98, idade 0, DebtRatio com renda nula)
  - [x] 1.2.9 1–2 frases de interpretação de negócio por gráfico
  - [x] 1.2.10 Exportar as figuras para `reports/figures/eda_*.png`
- [x] **T1.3** `tests/test_data.py` · 15 min
- **DoD:** notebook roda do início ao fim; 4+ figuras exportadas; 3 achados anotados para os slides.

#### E2. Features · 1 h 30
- [ ] **T2.1** `add_derived` (5 features da seção 6.3) · 20 min
- [ ] **T2.2** `QuantileClipper` (fit/transform) · 20 min
- [ ] **T2.3** `build_preprocessor(kind="tree"|"linear")` · 30 min
- [ ] **T2.4** `tests/test_features.py` (sem NaN, flags, percentis só do treino) · 20 min
- **DoD:** `preprocessor.fit_transform(X_train)` sem NaN; testes passando.

#### E3. Modelagem · 2 h
- [ ] **T3.1** Candidatos dummy, logreg e lgbm em Pipelines · 20 min
- [ ] **T3.2** CV 5-fold com AUC, Gini, KS, PR-AUC e Brier (média ± desvio) · 30 min
- [ ] **T3.3** Calibração isotônica + OOF via `cross_val_predict` · 25 min
- [ ] **T3.4** Varredura do corte por custo + comparação com 0,286 · 20 min
- [ ] **T3.5** Fit final + salvar o artefato com todos os metadados (5.3) · 15 min
- [ ] **T3.6** Salvar `reports/metrics/cv_results.json` · 10 min
- **DoD:** `python src/train.py` gera o `model.joblib` do zero em < 5 min; duas execuções dão métricas idênticas.

#### E4. Avaliação · 1 h 30
- [ ] **T4.1** Métricas no teste para os 3 modelos → JSON + tabela `.md` · 20 min
- [ ] **T4.2** Figuras: ROC, PR, KS, calibração · 30 min
- [ ] **T4.3** Curva de lucro com as 4 marcações · 20 min
- [ ] **T4.4** Matriz de confusão + taxa de aprovação + inadimplência da carteira aprovada · 20 min
- **DoD:** 6 figuras e `test_results.json` gerados; o teste foi usado só aqui.

#### E5. Explicabilidade e equidade · 2 h
- [ ] **T5.1** SHAP summary e bar (amostra de 2.000) · 30 min
- [ ] **T5.2** `reason_codes` + mapa `REASON_TEXTS` · 30 min
- [ ] **T5.3** `fairness.py`: métricas por faixa + DI + equal opportunity · 40 min
- [ ] **T5.4** Figura de equidade + 1 parágrafo de interpretação · 20 min
- [ ] ⭐ **T5.5** Waterfall SHAP dos 3 perfis
- **DoD:** toda recusa no teste tem 3 motivos não vazios; `fairness.json` gerado.

#### E6. Inferência e demo · 2 h
- [ ] **T6.1** `predict.py` (load, validate, predict_one, 3 faixas) · 30 min
- [ ] **T6.2** `tests/test_predict.py` · 15 min
- [ ] **T6.3** Streamlit: aba Análise + perfis na sidebar · 45 min
- [ ] **T6.4** Streamlit: abas Desempenho, Explicabilidade e Equidade (lendo de `reports/`) · 20 min
- [ ] **T6.5** Calibrar os perfis para caírem em APROVAR/REVISAR/RECUSAR · 10 min
- **DoD:** `streamlit run app/streamlit_app.py` abre sem erro e os 3 perfis dão as 3 decisões.

#### E7. Entrega · 2 h 30
- [ ] **T7.1** `README.md`: objetivo, 4 comandos (instalar, baixar, treinar, demo), estrutura · 20 min
- [ ] **T7.2** `relatorio.md` (estrutura da seção 13.1) · 1 h
- [ ] **T7.3** Slides (seção 13.2) · 50 min
- [ ] **T7.4** Gravar vídeo de backup da demo (1–2 min) · 10 min
- [ ] **T7.5** Ensaio cronometrado · 10 min
- **DoD:** checklist da seção 14 completo.

**Distribuição sugerida:**
| Dia | Épicos |
|---|---|
| Dia 1 | E0, E1, E2, E3, E4 |
| Dia 2 | E5, E6, E7 |

---

### FASE 2: Rigor acadêmico (⬜ opcional)

#### E8. Robustez estatística
- [ ] **T8.1** IC 95% da AUC, KS e lucro no teste por bootstrap (1.000 reamostragens)
- [ ] **T8.2** Bootstrap pareado lgbm × logreg (Q1): o IC da diferença contém 0?
- [ ] **T8.3** Ablação: sem features derivadas; sem `idade` (Q5)
- [ ] **T8.4** Sensibilidade da matriz de custos: perda de −3, −5 e −10 (Q6)
- [ ] **T8.5** Curva de aprendizado (AUC × tamanho do treino)

#### E9. Modelos adicionais
- [ ] **T9.1** XGBoost com `scale_pos_weight`
- [ ] **T9.2** Optuna (30 trials, CV 5-fold) no LightGBM
- [ ] **T9.3** Calibração Platt × isotônica (Brier)
- [ ] **T9.4** Experimento com SMOTE dentro do `imblearn.Pipeline`, para demonstrar o efeito na calibração

#### E10. Scorecard tradicional
- [ ] **T10.1** Binning ótimo + WoE + IV (`optbinning`)
- [ ] **T10.2** Logística sobre WoE → tabela de pontos
- [ ] **T10.3** Escala de score: `score = offset + fator·ln(odds)`, base 600 em odds 50:1, PDO = 20, limitada a 0–1000
- [ ] **T10.4** Score exibido na demo

#### E11. Generalização e equidade ampliada
- [ ] **T11.1** Replicar o pipeline no UCI *Default of Credit Card Clients*
- [ ] **T11.2** Equidade por sexo no UCI
- [ ] **T11.3** Contrafactual simples na demo ("o que mudaria a decisão")

#### E12. Documentação acadêmica
- [ ] **T12.1** `model_card.md` (uso pretendido, dados, métricas, limitações, equidade)
- [ ] **T12.2** Relatório em formato de artigo (ABNT ou SBC, conforme a disciplina)

---

### FASE 3: Rumo à produção (⬜ opcional)

#### E13. Empacotamento
- [ ] **T13.1** Converter `src/` no pacote `scorewise` com `pyproject.toml`
- [ ] **T13.2** Configuração por variáveis de ambiente (`pydantic-settings`)
- [ ] **T13.3** Logging estruturado (JSON)

#### E14. API
- [ ] **T14.1** `schemas.py` (Pydantic, com faixas validadas)
- [ ] **T14.2** Rotas da seção 8.2 usando `predict.py`
- [ ] **T14.3** Log de auditoria (8.3)
- [ ] **T14.4** `tests/test_api.py` com `TestClient`
- [ ] **T14.5** Teste de carga simples (p95 < 200 ms)

#### E15. Contêiner e CI
- [ ] **T15.1** `Dockerfile` *multi-stage* (usuário não root)
- [ ] **T15.2** `docker-compose.yml` (API + Streamlit + MLflow)
- [ ] **T15.3** GitHub Actions: ruff → pytest → docker build

#### E16. MLOps
- [ ] **T16.1** MLflow tracking em `train.py`
- [ ] **T16.2** Model Registry (Staging → Production)
- [ ] ⭐ **T16.3** DVC para `data/raw`

#### E17. Monitoramento
- [ ] **T17.1** `monitoring.py`: PSI do score e CSI por variável
- [ ] **T17.2** Demonstração simulando *drift* (perturbar a renda no teste) e mostrando o PSI subir
- [ ] **T17.3** Política de retreino e de champion/challenger documentada

---

## 13. Entregáveis acadêmicos

### 13.1 Estrutura do `relatorio.md`
1. Resumo (5–8 linhas, com os números finais)
2. Introdução e motivação
3. Fundamentação (risco de crédito, métricas, explicabilidade, equidade)
4. Dados (dicionário, qualidade, tratamento)
5. Metodologia (partição, pipeline, modelos, calibração, corte por custo)
6. Resultados (tabela comparativa, figuras, Q1–Q4)
7. Explicabilidade e equidade
8. Discussão ética e regulatória
9. Limitações (seção 15)
10. Trabalhos futuros (F2 e F3)
11. Conclusão
12. Referências

### 13.2 Slides (12)
| # | Slide | Fonte |
|---|---|---|
| 1 | ScoreWise: problema e proposta | 1.1–1.2 |
| 2 | Dados e definição de inadimplência | 2.2, 3.1 |
| 3 | 3 achados da EDA | E1 |
| 4 | Pipeline sem vazamento | 2.4, 5.1 |
| 5 | Modelos e tabela comparativa | E4 |
| 6 | Calibração: por que importa | `calibration.png` |
| 7 | Corte por custo (teoria 0,286 × prática) | `profit_curve.png` |
| 8 | SHAP global e reason codes | E5 |
| 9 | Equidade e LGPD | 6.7, 7 |
| 10 | **Demo ao vivo** | E6 |
| 11 | Limitações e roadmap F2/F3 | 12, 15 |
| 12 | Conclusão | |

### 13.3 Referências base
- Thomas, Edelman & Crook. *Credit Scoring and Its Applications*. SIAM.
- Siddiqi, N. *Credit Risk Scorecards*. Wiley.
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
- Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
- Kaggle (2011). *Give Me Some Credit*.
- Brasil. Lei 13.709/2018 (LGPD).

---

## 14. Checklist final (DoD do MVP)

- [ ] `python src/train.py` gera o `model.joblib` do zero, com resultado reprodutível
- [ ] `python src/evaluate.py` gera todas as figuras e JSONs
- [ ] `pytest` passa
- [ ] `streamlit run app/streamlit_app.py` funciona e os 3 perfis dão as 3 decisões
- [ ] Todo número dos slides e do relatório existe em `reports/metrics/`
- [ ] Premissas de custo e limitações explícitas no relatório
- [ ] README com instruções reproduzíveis
- [ ] Vídeo de backup gravado e apresentação ensaiada

---

## 15. Limitações declaradas

| ID | Limitação | Consequência | Trabalho futuro |
|---|---|---|---|
| L1 | Sem coluna temporal | Não avalia estabilidade no tempo | Dataset com data (Home Credit, base própria) |
| L2 | Sem propostas recusadas | Sem reject inference empírico | Base com histórico de recusas |
| L3 | Custos assumidos | Lucro é ilustrativo | Valor, prazo e taxa reais |
| L4 | Equidade só por idade | Sem análise de sexo/região | UCI (E11) ou base própria |
| L5 | Dados de outro país e época | Distribuições diferentes do Brasil atual | Bureaus brasileiros / open finance |
| L6 | Protótipo | Sem segurança, SLA ou auditoria reais | Fase 3 |

---

## 16. Glossário

| Termo | Definição |
|---|---|
| PD | Probabilidade de inadimplência |
| AUC | Probabilidade de o modelo ordenar um mau acima de um bom |
| Gini | 2·AUC − 1 |
| KS | Maior distância entre as distribuições acumuladas de score de bons e maus |
| Brier | Erro quadrático médio das probabilidades |
| Calibração | Ajuste para que uma PD de 10% corresponda a ~10% de maus na realidade |
| OOF | *Out-of-fold*: predição feita por um modelo que não viu aquela linha no treino |
| SHAP | Método que atribui a cada variável sua contribuição para uma predição |
| Reason code | Motivo de decisão em linguagem simples, derivado do SHAP |
| Disparate impact | Razão entre as taxas de aprovação de dois grupos |
| Equal opportunity | Igualdade da taxa de aprovação entre bons pagadores de grupos diferentes |
| WoE / IV | *Weight of Evidence* / *Information Value*: transformação e poder de separação por faixa, usados em scorecards |
| PDO | *Points to Double the Odds*: quantos pontos de score dobram a chance de ser bom pagador |
| PSI / CSI | Índices de estabilidade da população (score) e por variável |
| Reject inference | Técnicas para corrigir o viés de treinar só com quem foi aprovado |
| Champion/challenger | Modelo atual × modelo candidato, comparados em produção |
