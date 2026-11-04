# ScoreWise: modelo explicável de aprovação de crédito

> Todos os números deste relatório vêm de execuções registradas em `reports/metrics/` (`cv_results.json`, `test_results.json`, `explain.json`, `fairness.json`). Dados: *Give Me Some Credit* (SHA-256 do CSV: `1bd46da4…b24ac`). Semente global: 42. Modelo v0.1.0.

## 1. Resumo

O ScoreWise estima a probabilidade de inadimplência (PD) de uma proposta de crédito, decide entre aprovar, revisar manualmente ou recusar, e explica a decisão com três motivos em português. Usamos o dataset público *Give Me Some Credit* (149.391 propostas após remover 609 duplicadas; 6,70% de maus pagadores). O modelo final é um LightGBM com `class_weight`, calibrado por regressão isotônica, com o corte escolhido pelo lucro esperado. No conjunto de teste (29.879 propostas, usado uma única vez) obteve AUC 0,8648, Gini 0,7296 e KS 0,5815, e a calibração reduziu o Brier de 0,1363 para 0,0489. O corte por custo (0,270) ficou próximo do teórico (0,2857) e rendeu 0,6753 de lucro por proposta, contra 0,6384 no corte padrão de 0,5 e 0,5980 de aprovar todos (premissas de custo ilustrativas). A variável mais importante é a utilização do limite, seguida do total de atrasos e da idade. Há diferença de tratamento por idade (bons pagadores com menos de 30 anos são aprovados em 92,4% dos casos, contra 98,7% dos com 60+), discutida na seção 7. O projeto é um protótipo acadêmico com limitações declaradas na seção 9.

## 2. Introdução e motivação

Instituições financeiras decidem milhares de propostas com custos assimétricos: aprovar um mau pagador custa mais que recusar um bom. Além do desempenho, a decisão precisa ser explicável (LGPD, art. 20: direito à revisão de decisões automatizadas e a informações sobre os critérios; *conferir a norma vigente*) e auditável quanto a vieses. O objetivo é construir um MVP que una (i) um modelo preditivo bem avaliado, (ii) uma regra de decisão baseada em custo, (iii) explicações por proposta e (iv) uma análise de equidade por faixa etária, com demonstração ao vivo.

Perguntas de pesquisa: **Q1** o *gradient boosting* supera a regressão logística? **Q2** escolher o corte pelo custo gera mais valor que 0,5 ou que o corte por F1? **Q3** quais variáveis concentram o poder de decisão? **Q4** o modelo trata as faixas etárias de forma equitativa?

## 3. Fundamentação

- **PD calibrada:** uma PD de 10% deve corresponder a cerca de 10% de maus na realidade. Modelos treinados com `class_weight` deslocam as probabilidades para cima, por isso a calibração é obrigatória (Niculescu-Mizil & Caruana, 2005).
- **Métricas:** AUC (probabilidade de ordenar um mau acima de um bom), Gini = 2·AUC − 1, KS (maior distância entre as distribuições acumuladas de bons e maus), PR-AUC (relevante com classe rara) e Brier (erro quadrático médio da probabilidade).
- **Explicabilidade:** SHAP (Lundberg & Lee, 2017) atribui a cada variável sua contribuição para uma predição; os motivos da decisão são derivados dele.
- **Equidade:** *disparate impact* (razão entre taxas de aprovação, com alerta abaixo de 0,8) e *equal opportunity* (igualdade da taxa de aprovação entre bons pagadores; Hardt et al., 2016).

## 4. Dados

**Fonte:** `cs-training.csv` (150.000 linhas, 11 colunas e índice). O `cs-test.csv` não tem rótulo e não foi usado. Alvo: atraso ≥ 90 dias nos 2 anos seguintes (`SeriousDlqin2yrs`).

**Dicionário:** idade, renda mensal, razão dívida/renda, utilização do limite, linhas de crédito abertas, financiamentos imobiliários, dependentes e três contagens de atraso (30–59, 60–89, 90+ dias). Não existem no dataset: data da proposta, valor, prazo, taxa, sexo e região.

**Qualidade (medida na EDA):**

| Problema | Medido | Tratamento |
|---|---|---|
| Duplicadas | 609 (0,41%) | removidas antes do split |
| `MonthlyIncome` nula | 19,82% | mediana + flag `renda_ausente` (no Pipeline) |
| `NumberOfDependents` nulo | 2,62% | mediana (no Pipeline) |
| `age = 0` | 1 linha | vira nulo, depois mediana |
| Códigos 96/98 nos atrasos | 269 linhas (nas 3 colunas juntas) | viram nulo + flag `atraso_codigo_especial` |
| `DebtRatio` com renda nula | mediana 1.198 contra 0,30 com renda informada | flag de renda + clip p1–p99 |
| `RevolvingUtilization…` > 1 | 3.321 linhas, máximo 50.708 | clip p1–p99 |

As linhas com código 96/98 têm taxa de mau de 54,6% (contra 6,6% no restante), por isso viram uma feature. **Achados da EDA:** (1) a inadimplência cai de 11,9% (até 29 anos) para 3,1% (60+); (2) um único atraso de 90+ dias leva o risco de 4,6% para 33,7%; (3) a relação com a utilização do limite é não linear (1,3% a 2,6% nos 5 primeiros decis, 16,4% e 23,5% nos dois últimos).

## 5. Metodologia

**Partição.** Após a limpeza estática e a remoção de duplicadas, o dataset (149.391 linhas) é dividido em 80% treino (119.512; 6,6997% de maus) e 20% teste (29.879; 6,7004%), estratificado, seed 42. O teste só é lido por `evaluate.py`, `explain.py` e `fairness.py`, nunca no treino ou na escolha de corte. A divisão é estratificada, e não temporal, porque o dataset não tem data (L1).

**Prevenção de vazamento.** Só entram antes do split transformações que não aprendem parâmetros (duplicadas, `age = 0` e 96/98 viram nulo). Mediana, percentis do clip e padronização ficam dentro do `Pipeline` e são ajustados só no treino de cada fold. O `QuantileClipper` aprende p1 e p99 no `fit`, e há teste unitário garantindo que os limites não mudam ao transformar dados novos.

**Features.** 10 originais mais derivadas: total de atrasos, indicador de atraso grave, renda per capita, dívida mensal (razão × renda), `renda_ausente` e `atraso_codigo_especial`. Ao todo, 16 colunas chegam ao modelo.

**Modelos.** Dummy (prior, piso de referência), regressão logística (`class_weight="balanced"`, com `StandardScaler`) e LightGBM (500 árvores, taxa 0,03, 31 folhas, `class_weight="balanced"`). Não usamos SMOTE (distorce probabilidades) e a seleção é por AUC médio na CV 5-fold estratificada.

**Calibração e corte.** Sobre as probabilidades *out-of-fold* da CV (cada uma vinda de um modelo que não viu a linha), ajustamos uma `IsotonicRegression`. O corte é escolhido varrendo t de 0,01 a 0,99 (passo 0,005) sobre as OOF calibradas, maximizando o lucro esperado da matriz de custos. Aprova-se se PD < corte. Há uma faixa de **revisão humana** de ±0,05 em torno do corte, como resposta prática ao art. 20 da LGPD.

**Matriz de custos (premissas ilustrativas, não dados reais).**

| | Real: bom | Real: mau |
|---|---|---|
| Aprovar | +1 | −5 |
| Recusar | −1 | 0 |

O corte teórico com PD calibrada é p < 2/7 ≈ 0,2857. Dados reais exigiriam valor, prazo e taxa de cada empréstimo.

## 6. Resultados

**CV 5-fold (treino), média ± desvio:**

| Modelo | AUC | Gini | KS | PR-AUC | Brier (cru) |
|---|---|---|---|---|---|
| Dummy | 0,5000 ± 0,0000 | 0,0000 | 0,0000 | 0,0670 | 0,0625 |
| Logística | 0,8579 ± 0,0023 | 0,7158 ± 0,0046 | 0,5606 ± 0,0071 | 0,3818 ± 0,0058 | 0,1439 ± 0,0011 |
| LightGBM | 0,8634 ± 0,0025 | 0,7268 ± 0,0050 | 0,5695 ± 0,0093 | 0,3994 ± 0,0042 | 0,1313 ± 0,0005 |

**Teste (n = 29.879):**

| Modelo | AUC | Gini | KS | PR-AUC | Brier cru | Brier calibrado |
|---|---|---|---|---|---|---|
| Dummy | 0,5000 | 0,0000 | 0,0000 | 0,0670 | 0,0625 | 0,0625 |
| Logística | 0,8590 | 0,7181 | 0,5655 | 0,3908 | 0,1459 | 0,0494 |
| LightGBM | 0,8648 | 0,7296 | 0,5815 | 0,4037 | 0,1363 | 0,0489 |

Figuras: `roc.png`, `pr.png`, `ks.png`, `calibration.png`, `profit_curve.png`, `confusion.png` (em `reports/figures/`).

**Q1: boosting × logística.** O LightGBM é melhor em todos os 5 folds (diferença média de AUC de +0,0055, desvio 0,0021) e no teste (+0,0058). A diferença é pequena, porém consistente. Não calculamos intervalo de confiança por bootstrap (fica na F2), então **não afirmamos significância estatística**. A logística, muito mais simples, chega perto: o ganho de lucro do LightGBM sobre ela é de 0,6753 contra 0,669 por proposta no teste.

**Q2: corte por custo.** O corte empírico (0,270, escolhido nas OOF do treino) ficou perto do teórico (0,2857), indício de boa calibração. No teste:

| Regra | Corte | Lucro/proposta | Aprovação | Inadimplência dos aprovados |
|---|---|---|---|---|
| Padrão 0,5 | 0,500 | 0,6384 | 98,18% | 5,71% |
| F1 máximo | 0,210 | 0,6645 | 90,79% | 3,44% |
| **Custo** | **0,270** | **0,6753** | 93,70% | 4,05% |
| Teórico | 0,2857 | 0,6753 | 93,70% | 4,05% |

O corte por custo supera o 0,5 em 5,8% e o de F1 em 1,6%. Custo e teórico empatam porque a isotônica forma patamares e nenhuma PD cai entre 0,270 e 0,2857. Para referência, aprovar todos rende 0,5980 e recusar todos, −0,9330. O ganho do modelo sobre "aprovar todos" é de +0,0773 por proposta (+12,9%) **sob estas premissas**, e é modesto. No corte escolhido: 868 maus e 1.015 bons recusados, 1.134 maus e 26.862 bons aprovados. A inadimplência da carteira aprovada é 4,05%, contra 6,70% da população.

**Calibração.** O modelo cru, treinado com `class_weight`, superestima fortemente a PD (nos primeiros decis a PD crua média ia de ~3% a ~24%, para taxas observadas de ~0,6% a ~2%; ver `calibration.png`). Após a isotônica, a curva fica sobre a diagonal e o Brier cai de 0,1363 para 0,0489. A AUC praticamente não muda (0,8648 → 0,8647).

**Reprodutibilidade.** `python src\train.py` leva cerca de 20 s, e duas execuções (e uma reexecução do zero, com o modelo apagado) geram o mesmo `cv_results.json` (mesmo SHA-256).

## 7. Explicabilidade e equidade

**Q3: variáveis mais importantes (|SHAP| médio, amostra de 2.000 do teste):** utilização do limite (0,729), total de atrasos (0,517), idade (0,299), linhas abertas (0,194), razão dívida/renda (0,140) e renda mensal (0,125). A flag `atraso_codigo_especial` pesa pouco na média, mas desloca o risco em cerca de +1,3 log-odds quando ativa. O SHAP é calculado sobre o modelo cru, em log-odds, porque a calibração é monotônica e não altera a ordem dos fatores.

**Reason codes.** Cada decisão traz até 3 motivos, só de fatores que *aumentam* o risco, traduzidos por um mapa de textos. Flags binárias só entram quando ativas (o SHAP de uma flag = 0 pode ser positivo em relação à média, e dizer "renda não informada" a quem informou seria incorreto; há teste de regressão). Das 1.883 recusas no teste, todas têm 3 motivos.

**Q4: equidade por faixa etária (corte 0,270):**

| Faixa | n | Mau real | Aprovação | DI | TPR (bom aprovado) |
|---|---|---|---|---|---|
| < 30 | 1.781 | 11,90% | 86,81% | 0,890 | 92,35% |
| 30–44 | 7.869 | 9,07% | 90,99% | 0,933 | 94,69% |
| 45–59 | 10.698 | 7,16% | 93,43% | 0,958 | 96,05% |
| 60+ | 9.531 | 3,25% | 97,52% | 1,000 | 98,67% |

O DI mínimo (0,890) fica acima do limite de 0,8, então nenhuma faixa dispara o alerta (a regra dos 4/5 é uma heurística de origem estadunidense, e não norma brasileira). A diferença de aprovação é em boa parte explicada pelo risco observado (a inadimplência dos menores de 30 é 3,7 vezes a dos 60+). Porém a *equal opportunity* revela uma disparidade além do risco médio do grupo: entre os **bons pagadores**, 7,65% dos menores de 30 são recusados contra 1,33% dos 60+ (diferença de 6,3 p.p.). A idade é a 3ª variável mais importante do modelo. A quantificação do efeito de retirá-la (ablação) fica para a F2. A faixa < 30 tem só 1.781 propostas, então suas taxas têm mais incerteza, e não calculamos intervalos de confiança.

## 8. Discussão ética e regulatória

*Toda citação normativa abaixo deve ser conferida na norma vigente antes da entrega.*

- **LGPD, art. 20:** direito à revisão de decisões automatizadas e a informações sobre os critérios. Os reason codes atendem parcialmente à informação, e a faixa de revisão humana atende parcialmente à revisão. Isso não equivale a conformidade jurídica.
- **Idade como variável:** é preditiva e sensível. O motivo "Perfil etário associado a maior risco" pode aparecer nas explicações, o que merece cuidado.
- **Proxies:** renda e dependentes podem carregar informação socioeconômica correlacionada a grupos protegidos. Sexo e região não existem no dataset, então não foram analisados (L4).
- **Reject inference (teórico):** o modelo só vê quem foi aprovado no passado (viés de seleção). Técnicas a considerar: *reweighting/augmentation* e *parceling*. Não implementadas, pois o dataset não tem recusados (L2).
- **Feedback loop:** em produção, quem é recusado nunca gera rótulo, o que reforça o viés ao longo do tempo.

## 9. Limitações

| ID | Limitação | Consequência |
|---|---|---|
| L1 | Sem coluna temporal; partição estratificada | Não avalia estabilidade no tempo |
| L2 | Sem propostas recusadas | Sem reject inference empírico |
| L3 | Custos assumidos (+1 / −5 / −1 / 0) | O lucro é ilustrativo; sem análise de sensibilidade |
| L4 | Equidade só por idade | Sem análise de sexo ou região |
| L5 | Dados de outro país e época (2011) | Distribuições diferentes do Brasil atual |
| L6 | Protótipo | Sem segurança, SLA, auditoria ou versionamento reais |

Limitações adicionais: (i) o corte é escolhido sobre OOF calibradas pelo mesmo calibrador ajustado nelas (leve otimismo; o teste, intocado, confirma o resultado); (ii) sem IC por bootstrap, a comparação entre modelos é descritiva; (iii) as colunas de atraso são colineares, então não interpretamos coeficientes individuais da logística; (iv) o `DebtRatio` mistura razão e valor absoluto quando a renda é nula; (v) a demo não permite informar o código 96/98.

## 10. Trabalhos futuros

- **Fase 2 (rigor acadêmico):** IC 95% por bootstrap e teste pareado (Q1); ablação sem `idade` (Q5); sensibilidade da matriz de custos com perda de −3, −5 e −10 (Q6); XGBoost e Optuna; calibração Platt × isotônica; SMOTE dentro do Pipeline para demonstrar o efeito na calibração; scorecard tradicional (WoE/IV e escala de pontos); replicação no dataset UCI *Default of Credit Card Clients* (equidade por sexo); *model card*.
- **Fase 3 (produção):** API (FastAPI) e contêiner (Docker), log de auditoria, rastreamento de experimentos (MLflow), monitoramento de estabilidade (PSI/CSI), política de retreino e *champion/challenger*, CI.
- **Dados:** dataset com data (split temporal), histórico de recusas (reject inference), valor/prazo/taxa reais (LGD/EAD) e bureaus brasileiros.

## 11. Conclusão

Um modelo de gradient boosting calibrado, com corte baseado em custo e explicações por SHAP, alcança AUC 0,865 no teste e PD bem calibrada (Brier 0,049), com um corte empírico (0,270) consistente com a teoria (0,2857). Sob premissas de custo ilustrativas, o corte por custo supera o corte padrão e o de F1, mas o ganho sobre aprovar todos é modesto (+12,9%), e o LightGBM supera a logística por margem pequena. A análise de equidade mostra que o modelo tem DI aceitável por faixa etária, porém trata menos favoravelmente os bons pagadores mais jovens, um ponto que exige atenção ética e é o principal encaminhamento para a Fase 2.

## 12. Referências

- Thomas, Edelman & Crook. *Credit Scoring and Its Applications*. SIAM.
- Siddiqi, N. *Credit Risk Scorecards*. Wiley.
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
- Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
- Kaggle (2011). *Give Me Some Credit*.
- Brasil. Lei 13.709/2018 (LGPD).
