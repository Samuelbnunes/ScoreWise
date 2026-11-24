# ScoreWise: modelo explicável de aprovação de crédito

> Todos os números deste relatório vêm de execuções registradas em `reports/metrics/` (`cv_results.json`, `test_results.json`, `explain.json`, `fairness.json`, `stats.json`). Dados: *Give Me Some Credit* (SHA-256 do CSV: `1bd46da4…b24ac`). Semente global: 42. Modelo v0.2.0.

## 1. Resumo

O ScoreWise estima a probabilidade de inadimplência (PD) de uma proposta de crédito, decide entre aprovar, encaminhar para revisão humana ou recusar, e explica a decisão com até três motivos distintos em português. Usamos o dataset público *Give Me Some Credit* (149.352 propostas após remover 648 duplicadas; 6,70% de maus pagadores).

**O modelo.** O modelo final é um LightGBM com três características:
- **restrições monotônicas**: mais atraso, uso do limite ou endividamento nunca reduz o risco, e mais renda nunca o aumenta;
- treinado **sem a idade** e **sem `class_weight`**;
- com calibração de Platt.

O corte de aprovação é escolhido pelo lucro esperado. A revisão humana fica na faixa de PD em que ela vale mais que a decisão automática: de 20,7% a 34,1%.

**Resultados no teste** (29.871 propostas, usado uma única vez):
- AUC 0,8644 (IC 95% por bootstrap: 0,8561 a 0,8724), Gini 0,7288, KS 0,5673;
- Brier 0,0483, com a probabilidade já calibrada de origem;
- o corte por custo (0,260) rendeu 0,6764 de lucro por proposta, contra 0,6457 no corte padrão de 0,5 e 0,5979 ao aprovar todos. As premissas de custo são ilustrativas, e o ganho sobre aprovar todos varia de +4,3% a +93,5% conforme a perda por mau aprovado (Q6);
- na decisão em 3 faixas, 91,8% das propostas são aprovadas automaticamente, 3,9% vão para revisão e 4,4% são recusadas.

**Comparação com a logística.** A superioridade em AUC é estatisticamente significativa, porém pequena: ΔAUC +0,0070, IC 95% de 0,0038 a 0,0102. A diferença de lucro não é significativa.

**Explicação e equidade.** As variáveis mais importantes são a utilização do limite e o total de atrasos. Entre os bons pagadores, os menores de 30 anos são aprovados em 92,7% dos casos, contra 98,3% dos com 60+ (seção 7). Tirar a idade do modelo reduziu essa diferença de 6,6 para 5,5 pontos percentuais de forma significativa, mas não a eliminou. O projeto é um protótipo acadêmico, com limitações declaradas na seção 9.

## 2. Introdução e motivação

Instituições financeiras decidem milhares de propostas com custos assimétricos: aprovar um mau pagador custa mais que recusar um bom. Além do desempenho, a decisão precisa ser:
- **explicável**: a LGPD, no art. 20, prevê o direito à revisão de decisões automatizadas e a informações sobre os critérios (*conferir a norma vigente*);
- **coerente**: a mesma pessoa não pode ser aprovada com renda menor e revisada com renda maior;
- **auditável** quanto a vieses.

O objetivo é construir um MVP que una:
1. um modelo preditivo bem avaliado;
2. uma regra de decisão baseada em custo;
3. explicações por proposta;
4. uma análise de equidade por faixa etária, com demonstração ao vivo.

**Perguntas de pesquisa:**
- **Q1:** o *gradient boosting* supera a regressão logística?
- **Q2:** escolher o corte pelo custo gera mais valor que o corte de 0,5 ou o corte por F1?
- **Q3:** quais variáveis concentram o poder de decisão?
- **Q4:** o modelo trata as faixas etárias de forma equitativa?
- **Q5:** usar a idade aumenta a disparidade, e o que se ganha com ela?
- **Q6:** a decisão é sensível às premissas de custo?

## 3. Fundamentação

- **PD calibrada:** uma PD de 10% deve corresponder a cerca de 10% de maus na realidade. Pesos de classe (`class_weight`) deslocam as probabilidades para cima e obrigam a recalibrar (Niculescu-Mizil & Caruana, 2005). Por isso treinamos sem eles e só ajustamos o pouco que falta.
- **Métricas:**
  - AUC: probabilidade de o modelo ordenar um mau acima de um bom;
  - Gini = 2·AUC − 1;
  - KS: maior distância entre as distribuições acumuladas de bons e maus;
  - PR-AUC: relevante com classe rara;
  - Brier: erro quadrático médio da probabilidade;
  - ECE: diferença média entre a PD prevista e a taxa observada, por faixas.
- **Restrições monotônicas:** obrigam o modelo a respeitar o sentido esperado de uma variável (por exemplo, o risco não diminui quando os atrasos aumentam). Isso torna as decisões e as explicações coerentes.
- **Explicabilidade:** SHAP (Lundberg & Lee, 2017) atribui a cada variável sua contribuição para uma predição. Os motivos da decisão são derivados dele.
- **Equidade:** *disparate impact* (razão entre taxas de aprovação, com alerta abaixo de 0,8) e *equal opportunity* (igualdade da taxa de aprovação entre bons pagadores; Hardt et al., 2016).

## 4. Dados

**Fonte:** `cs-training.csv` (150.000 linhas, 11 colunas e índice). O `cs-test.csv` não tem rótulo e não foi usado. Alvo: atraso de 90 dias ou mais nos 2 anos seguintes (`SeriousDlqin2yrs`).

**Dicionário:** idade, renda mensal, razão dívida/renda, utilização do limite, linhas de crédito abertas, financiamentos imobiliários, dependentes e três contagens de atraso (30–59, 60–89 e 90+ dias). Não existem no dataset: data da proposta, valor, prazo, taxa, sexo e região.

**Qualidade (medida na EDA, `notebooks/01_eda.ipynb`):**

| Problema | Medido | Tratamento |
|---|---|---|
| Duplicadas | 648 (0,43%) | Removidas antes do split |
| `MonthlyIncome` nula | 19,82% | Mediana + flag `renda_ausente` (no Pipeline) |
| `MonthlyIncome` = 0 | 1.634 linhas (1,09%); `DebtRatio` mediana 930, 93,1% acima de 1 | Tratada como **renda não informada** (com renda 0, o `DebtRatio` é dívida em valor, como no grupo de renda nula) |
| `NumberOfDependents` nulo | 2,62% | Mediana (no Pipeline) |
| `age = 0` | 1 linha | Vira nulo |
| Códigos 96/98 nos atrasos | 269 linhas (nas 3 colunas juntas) | Viram nulo + flag `atraso_codigo_especial` |
| `DebtRatio` com renda ausente | Mediana 1.188,5, contra 0,29 com renda informada | Flag de renda + clip p1–p99; na demo, o campo pede o valor da dívida |
| `RevolvingUtilization…` > 1 | 3.321 linhas, máximo 50.708 | Clip p1–p99 |
| Utilização = 0 | 10.878 linhas, com 2,94% de maus (contra 1,28% em (0; 0,01]) | Flag `sem_uso_rotativo` |

As linhas com código 96/98 têm taxa de mau de 54,6%, contra 6,6% no restante, por isso viram uma feature. Após a limpeza, a renda ausente soma 20,62% das linhas.

**Achados da EDA:**
1. A inadimplência cai de 11,9% (até 29 anos) para 3,1% (60+).
2. Um único atraso de 90+ dias leva o risco de 4,6% para 33,7%.
3. A relação com a utilização do limite é não linear: de 1,3% a 2,6% nos 5 primeiros decis, e 16,4% e 23,5% nos dois últimos.
4. Quem não usa crédito rotativo tem mais que o dobro do risco de quem usa até 1% do limite.

## 5. Metodologia

**Partição.** Após a limpeza estática e a remoção de duplicadas, o dataset (149.352 linhas) é dividido em 80% treino (119.481; 6,70% de maus) e 20% teste (29.871; 6,70%), com estratificação e seed 42. O teste só é lido pelos scripts de avaliação (`evaluate.py`, `explain.py`, `fairness.py`, `stats.py`), nunca no treino nem na escolha do corte ou da faixa. A divisão é estratificada, e não temporal, porque o dataset não tem data (L1).

**Prevenção de vazamento.** Antes do split, só entram transformações que não aprendem parâmetros: remoção de duplicadas e conversão para nulo de `age = 0`, renda 0 e códigos 96/98. Mediana, percentis do clip e padronização ficam dentro do `Pipeline` e são ajustados só no treino de cada fold. O `QuantileClipper` aprende p1 e p99 no `fit`, e um teste unitário garante que os limites não mudam ao transformar dados novos.

**Features.** O modelo recebe 15 colunas:
- 9 originais (a idade fica de fora);
- total de atrasos, indicador de atraso grave e renda per capita;
- as flags `renda_ausente`, `atraso_codigo_especial` e `sem_uso_rotativo`.

A **idade não é usada pelo modelo**: é atributo sensível e serve só para auditar a equidade. A dívida mensal (razão × renda) também ficou de fora: ela cresce com a renda, o que conflita com a restrição de que mais renda não aumenta o risco, e retirá-la não reduziu a AUC na CV (0,8597 sem, 0,8595 com).

**Modelos.** Os três candidatos usam as mesmas 15 colunas e **nenhum peso de classe**:
- Dummy (prior), como piso de referência;
- regressão logística (com `StandardScaler`);
- LightGBM (500 árvores, taxa 0,03, 31 folhas, `min_child_samples` 100, subamostragem de 80% das linhas e colunas).

O LightGBM tem **restrições monotônicas** (método `intermediate`):
- crescentes em atrasos (30–59, 60–89, 90+, total, atraso grave, código especial), utilização do limite e razão dívida/renda;
- decrescentes em renda e renda per capita.

Não usamos SMOTE, porque distorce probabilidades. A seleção é por AUC médio na CV 5-fold estratificada.

As escolhas de configuração foram feitas **só na CV do treino**, por regras definidas antes:
- tirar a dívida mensal só se a perda de AUC fosse menor que 0,001;
- usar o método de monotonia de maior AUC;
- calibração por critério de ECE;
- novos hiperparâmetros só se ganhassem pelo menos 0,0025 de AUC.

Uma busca aleatória de 14 configurações de hiperparâmetros ganhou +0,0021, abaixo do mínimo, e por isso os parâmetros originais foram mantidos.

**Calibração.** Sobre as probabilidades *out-of-fold* da CV (cada uma vinda de um modelo que não viu a linha), comparamos três opções pela ordem de preferência: sem calibrador, Platt e isotônica. A regra exigia ECE ≤ 0,002 e PD média a até 0,2 p.p. da taxa real.
- Sem calibrador: ECE 0,0034, acima do limite.
- Platt: ECE 0,0019, adotado.
- Isotônica: descartada, porque gera PD em degraus.

**Corte e decisão.** O corte varre t de 0,01 a 0,99 (passo 0,005) sobre as OOF calibradas e maximiza o lucro esperado da matriz de custos. A decisão usa **três faixas**: aprovar, revisão humana ou recusar. A faixa de revisão é calculada, e não fixada: é o intervalo de PD em que o valor esperado de revisar supera o de decidir automaticamente, dado um revisor **ilustrativo**:
- custo de 0,2 por revisão (na unidade da matriz de custos);
- recusa 50% dos maus que revisa;
- aprova 80% dos bons que revisa.

Com esses valores, revisar vale a pena entre **20,7% e 34,1%** de PD. O corte teórico (28,6%) fica dentro dessa faixa. É a resposta prática ao direito de revisão do art. 20 da LGPD.

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
| Logística | 0,8511 ± 0,0075 | 0,7023 ± 0,0150 | 0,5443 ± 0,0149 | 0,3782 ± 0,0180 | 0,0507 ± 0,0009 |
| LightGBM | 0,8597 ± 0,0067 | 0,7193 ± 0,0134 | 0,5664 ± 0,0162 | 0,3933 ± 0,0178 | 0,0497 ± 0,0009 |

O desvio entre folds vem principalmente do 4º fold, mais difícil para todos os modelos (LightGBM 0,8477; nos outros, de 0,8619 a 0,8639).

**Teste (n = 29.871):**

| Modelo | AUC | Gini | KS | PR-AUC | Brier cru | Brier calibrado |
|---|---|---|---|---|---|---|
| Dummy | 0,5000 | 0,0000 | 0,0000 | 0,0670 | 0,0625 | 0,0625 |
| Logística | 0,8574 | 0,7148 | 0,5531 | 0,3960 | 0,0497 | 0,0497 |
| LightGBM | 0,8644 | 0,7288 | 0,5673 | 0,4199 | 0,0483 | 0,0483 |

Figuras em `reports/figures/`: `roc.png`, `pr.png`, `ks.png`, `calibration.png`, `profit_curve.png` e `confusion.png`.

**Q1: boosting × logística.** Bootstrap no teste (1.000 reamostragens, seed 42; `reports/metrics/stats.json`), IC 95% pelo método dos percentis:

| Modelo | AUC | KS | Lucro/proposta (corte por custo) |
|---|---|---|---|
| LightGBM | 0,8644 [0,8561; 0,8724] | 0,5673 [0,5511; 0,5869] | 0,6764 [0,6624; 0,6893] |
| Logística | 0,8574 [0,8490; 0,8660] | 0,5531 [0,5362; 0,5721] | 0,6740 [0,6601; 0,6874] |
| **Diferença pareada (LightGBM − logística)** | **+0,0070 [0,0038; 0,0102]** | **+0,0141 [0,0023; 0,0265]** | +0,0023 [−0,0024; 0,0071] |

O bootstrap **pareado** usa as mesmas reamostras para os dois modelos. Ele mostra que o LightGBM ordena melhor (os ICs de AUC e KS não contêm 0), mas por margem pequena, e que **a diferença de lucro não é significativa** (o IC contém 0). Na CV, o LightGBM vence nos 5 folds: diferença média +0,0085, desvio 0,0030; por fold, +0,0056, +0,0066, +0,0135, +0,0084 e +0,0085. As duas abordagens partem do mesmo ponto: as mesmas variáveis e nenhum peso de classe. O bootstrap captura apenas a variabilidade amostral do teste, não a do treino, porque o modelo é fixo.

**Q2: corte por custo.** O corte empírico foi 0,260, escolhido nas OOF do treino; o teórico é 0,2857. No teste:

| Regra | Corte | Lucro/proposta | Aprovação | Inadimplência dos aprovados |
|---|---|---|---|---|
| Padrão 0,5 | 0,500 | 0,6457 | 98,14% | 5,59% |
| F1 máximo | 0,210 | 0,6729 | 91,92% | 3,61% |
| **Custo** | **0,260** | **0,6764** | 93,67% | 4,03% |
| Teórico | 0,2857 | 0,6764 | 94,40% | 4,22% |

O corte por custo supera o de 0,5 em 4,8% e o de F1 em 0,5%. O corte por custo e o teórico empatam no lucro por coincidência, com decisões diferentes: o teórico recusa 62 maus a menos (−310) e 155 bons a menos (+310). Para referência, aprovar todos rende 0,5979 e recusar todos rende −0,9330. O ganho do modelo sobre "aprovar todos" é de +0,0785 por proposta (+13,1%) **sob estas premissas**.

No corte escolhido, foram recusados 875 maus e 1.015 bons, e aprovados 1.127 maus e 26.854 bons. A inadimplência da carteira aprovada é 4,03%, contra 6,70% da população.

**Decisão em 3 faixas (o que o produto faz).**

| Decisão | Propostas | % | Taxa de mau |
|---|---|---|---|
| Aprovar (PD < 20,7%) | 27.411 | 91,76% | 3,59% |
| Revisão humana (20,7% a 34,1%) | 1.156 | 3,87% | 28,29% |
| Recusar (PD > 34,1%) | 1.304 | 4,37% | 53,07% |

Com o revisor ilustrativo, o lucro esperado é de 0,6813 por proposta. Para referência:
- aprovar toda a faixa de revisão rende 0,6727;
- recusá-la inteira rende 0,6720;
- um revisor perfeito e sem custo renderia 0,7275.

A revisão só agrega valor se o analista souber algo que o modelo não sabe. A tabela de sensibilidade mostra como a faixa muda com o revisor (especificidade fixa em 80%):

| Custo da revisão | Maus recusados pelo analista | Faixa de revisão | % para revisão | Lucro esperado/proposta |
|---|---|---|---|---|
| 0,1 | 30% | 26,3% a 29,4% | 0,82% | 0,6768 |
| 0,1 | 50% | 17,2% a 36,6% | 5,75% | 0,6848 |
| 0,1 | 70% | 12,8% a 48,4% | 10,92% | 0,7054 |
| 0,2 | 30% | vazia | 0% | 0,6764 |
| **0,2** | **50%** | **20,7% a 34,1%** | **3,87%** | **0,6813** |
| 0,2 | 70% | 15,4% a 45,2% | 8,27% | 0,6963 |
| 0,5 | 30% | vazia | 0% | 0,6764 |
| 0,5 | 50% | vazia | 0% | 0,6764 |
| 0,5 | 70% | 23,1% a 35,5% | 3,19% | 0,6794 |

Com faixa vazia, a revisão nunca compensa e a decisão volta a ter 2 faixas no corte.

**Calibração.** Treinado sem pesos de classe, o modelo já sai quase calibrado: o Brier é 0,0483 antes e depois da calibração de Platt. Por decil, a PD crua média vai de 0,33% a 37,5%, para taxas observadas de 0,54% a 37,0% (`calibration.png`). A PD é contínua, sem degraus, e a AUC não muda (0,8644).

**Coerência das decisões.** Variamos uma entrada por vez nos três perfis da demo: renda de 250 a 20.000, utilização de 0,02 a 1,5, razão dívida/renda de 0 a 3 e atrasos de 0 a 7. Não houve nenhuma reversão de decisão (por exemplo, voltar de "revisar" para "aprovar" com mais dívida) e nenhum passo de PD no sentido contrário ao esperado. A renda 0 produz a mesma PD da renda não informada, e a idade não altera a PD. Há testes automatizados para essas propriedades (`tests/test_predict.py`).

**Reprodutibilidade.** `python src\train.py` leva cerca de 30 s. O `stats.py` retreina o modelo com o mesmo protocolo e confere, por asserção, que o corte e a AUC de cada fold são idênticos aos do `model.joblib` e do `cv_results.json`. São 61 testes automatizados (`pytest`), todos passando.

**Q5: e se a idade estivesse no modelo?** Treinamos o contrafactual, com a mesma configuração e protocolo e a idade como variável. Ele teve corte por custo de 0,270.

| Métrica (teste) | Modelo final (sem idade) | Com idade | Diferença (final − com idade) [IC 95%] |
|---|---|---|---|
| AUC | 0,8644 | 0,8673 | −0,0029 [−0,0053; −0,0010] |
| KS | 0,5673 | 0,5760 | −0,0088 [−0,0195; −0,0002] |
| Lucro/proposta | 0,6764 | 0,6790 | −0,0026 [−0,0058; 0,0006] |
| Brier calibrado | 0,0483 | 0,0482 | — |
| Aprovação / inadimplência dos aprovados | 93,67% / 4,03% | 93,80% / 4,02% | — |
| *Disparate impact* mínimo | 0,900 | 0,887 | — |
| TPR dos bons pagadores < 30 anos | 92,74% | 91,99% | — |
| TPR dos bons pagadores 60+ | 98,28% | 98,61% | — |
| Diferença de TPR (60+ − < 30) [IC 95%] | 5,54 p.p. [4,22; 6,93] | 6,62 p.p. [5,30; 8,04] | −1,08 p.p. [−1,68; −0,44] |

Usar a idade traria 0,0029 de AUC (pequeno, mas significativo) e **nenhum ganho detectável de lucro** (o IC contém 0), ao custo de aumentar a disparidade de forma significativa. Mesmo sem a idade, a diferença de TPR continua em 5,5 p.p.: parte do efeito vem de variáveis correlacionadas com a idade (renda, dependentes, linhas de crédito, histórico de atrasos). Retirar o atributo sensível não elimina a discriminação indireta. Ressalva: cada modelo usa o seu próprio corte por custo.

**Q6: sensibilidade da matriz de custos.** Não há retreino: a calibração não depende dos custos, o corte ótimo é recalculado nas OOF do treino e o lucro é medido no teste. Só a perda por mau aprovado varia; os demais valores seguem +1 / −1 / 0:

| Perda por mau aprovado | Corte teórico | Corte ótimo (OOF) | Lucro/proposta | Aprovar todos | Ganho s/ aprovar todos | Lucro mantendo 0,260 | Lucro com corte 0,5 | Aprovação | Inadimpl. aprovados |
|---|---|---|---|---|---|---|---|---|---|
| −3 | 0,400 | 0,400 | 0,7635 | 0,7319 | +4,3% | 0,7518 | 0,7554 | 96,62% | 4,88% |
| **−5** (base) | 0,2857 | 0,260 | 0,6764 | 0,5979 | +13,1% | 0,6764 | 0,6457 | 93,67% | 4,03% |
| −10 | 0,1667 | 0,160 | 0,5084 | 0,2628 | +93,5% | 0,4877 | 0,3714 | 89,61% | 3,26% |

A decisão **é sensível às premissas**: o corte ótimo vai de 0,400 a 0,160, e o valor do modelo sobre aprovar todos vai de +4,3% a +93,5%.
- Com perda baixa (−3), aprovar todos já é quase ótimo: o corte 0,5 rende 0,7554, contra 0,7635 do ótimo.
- Com perda alta (−10), o corte 0,5 perde 26,9% do lucro ótimo, e manter 0,260 perde 4,1%.
- Com PD calibrada e contínua, o corte ótimo acompanha o teórico nos três cenários (0,400 × 0,400; 0,260 × 0,286; 0,160 × 0,167).

Conclusão: o corte (e a faixa de revisão) tem de ser recalibrado quando a economia do produto muda, e todo valor de lucro deste relatório depende de premissas que não vêm de dados reais.

## 7. Explicabilidade e equidade

**Q3: variáveis mais importantes** (|SHAP| médio, amostra de 2.000 do teste): utilização do limite (0,773), total de atrasos (0,523), razão dívida/renda (0,195), linhas abertas (0,184), financiamentos imobiliários (0,160) e renda não informada (0,144). O SHAP é calculado sobre o modelo cru, em log-odds, porque a calibração é monotônica e não altera a ordem dos fatores.

**Reason codes.** Cada decisão traz até 3 motivos **distintos**, só de fatores que *aumentam* o risco, traduzidos por um mapa de textos. Regras:
- variáveis que dizem a mesma coisa são **agrupadas**: o SHAP de todas as de atraso é somado e vira um único motivo, com o texto do atraso mais grave presente; o mesmo vale para renda e renda per capita;
- flags binárias só entram quando ativas (dizer "renda não informada" a quem informou seria incorreto);
- com renda não informada, a dívida é descrita como valor;
- a utilização zero nunca é descrita como "uso elevado".

Das 1.304 recusas no teste, 1.279 têm 3 motivos, 24 têm 2 e 1 tem 1; todas têm ao menos um motivo. O primeiro motivo mais comum é "Histórico de atrasos superiores a 90 dias" (906 recusas), e "Uso elevado do limite de crédito disponível" aparece em 94% delas.

**Q4: equidade por faixa etária.** A idade não é variável do modelo, mas é usada aqui para auditar o resultado. Corte simples (0,260):

| Faixa | n | Mau real | Aprovação | DI | TPR (bom aprovado) |
|---|---|---|---|---|---|
| < 30 | 1.673 | 12,67% | 87,51% | 0,900 | 92,74% |
| 30–44 | 7.786 | 9,21% | 91,18% | 0,938 | 94,86% |
| 45–59 | 10.756 | 7,28% | 93,29% | 0,960 | 96,14% |
| 60+ | 9.656 | 3,00% | 97,18% | 1,000 | 98,28% |

O DI mínimo (0,900) fica acima do limite de 0,8, então nenhuma faixa dispara o alerta. A regra dos 4/5 é uma heurística de origem estadunidense, e não norma brasileira. A diferença de aprovação é em boa parte explicada pelo risco observado: a inadimplência dos menores de 30 é 4,2 vezes a dos 60+. Porém a *equal opportunity* revela uma disparidade além do risco médio do grupo: entre os **bons pagadores**, 7,26% dos menores de 30 são recusados, contra 1,72% dos 60+. A diferença é de 5,5 p.p. (IC 95% de 4,2 a 6,9).

Na decisão em 3 faixas, a aprovação automática vai de 83,8% (< 30) a 96,1% (60+). A faixa de revisão humana recebe 8,3% das propostas de menores de 30, contra 2,1% das de 60+. O DI da aprovação automática é 0,872. A faixa < 30 tem só 1.673 propostas, então suas taxas têm mais incerteza.

## 8. Discussão ética e regulatória

*Toda citação normativa abaixo deve ser conferida na norma vigente antes da entrega.*

- **LGPD, art. 20:** prevê o direito à revisão de decisões automatizadas e a informações sobre os critérios. Os reason codes atendem parcialmente à informação, e a faixa de revisão humana atende parcialmente à revisão. Isso não equivale a conformidade jurídica.
- **Idade:** é preditiva e sensível. Por isso **não é usada pelo modelo**. Usá-la e só escondê-la dos motivos tinha um efeito colateral: quando a idade era o fator principal, o cliente recebia um motivo substituto enganoso (por exemplo, "número de dependentes"). O contrafactual da Q5 mostra que usar a idade aumentaria a disparidade sem ganho detectável de lucro. Ela segue no formulário e nos dados só para auditar a equidade.
- **Coerência como requisito ético:** sem restrições monotônicas, a mesma pessoa podia ser aprovada com renda menor e revisada com renda maior, o que é difícil de justificar a um cliente ou a um regulador. Com as restrições, uma decisão pode ser explicada também como "o que mudaria a decisão".
- **Proxies:** renda e dependentes podem carregar informação socioeconômica correlacionada a grupos protegidos. Sexo e região não existem no dataset, então não foram analisados (L4).
- **Incentivo a omitir a renda:** quem não informa a renda recebe, em média, PD um pouco menor que quem informa uma renda média. Isso reflete os dados (renda não informada: 5,6% de maus; informada: 7,0%), mas cria um incentivo a omitir a informação (L7).
- **Reject inference (teórico):** o modelo só vê quem foi aprovado no passado (viés de seleção). Técnicas a considerar: *reweighting/augmentation* e *parceling*. Não foram implementadas, pois o dataset não tem recusados (L2).
- **Feedback loop:** em produção, quem é recusado nunca gera rótulo, o que reforça o viés ao longo do tempo.

## 9. Limitações

| ID | Limitação | Consequência |
|---|---|---|
| L1 | Sem coluna temporal; partição estratificada | Não avalia estabilidade no tempo |
| L2 | Sem propostas recusadas | Sem reject inference empírico |
| L3 | Custos e revisor assumidos (+1 / −5 / −1 / 0; revisão com custo 0,2 e acertos de 50% e 80%) | O lucro e a faixa de revisão são ilustrativos e sensíveis às premissas (Q6 e tabela de sensibilidade da revisão) |
| L4 | Equidade só por idade | Sem análise de sexo ou região |
| L5 | Dados de outro país e época (2011) | Distribuições diferentes do Brasil atual |
| L6 | Protótipo | Sem segurança, SLA, auditoria ou versionamento reais |
| L7 | Renda não informada recebe PD um pouco menor que uma renda média (perfil Limítrofe: 24,1% sem renda, contra 25,7% com renda de 3.500) | Incentivo a omitir a renda; em produção, exigiria regra de negócio ou comprovação |

**Limitações adicionais:**
1. O corte é escolhido sobre OOF calibradas pelo mesmo calibrador ajustado nelas. Há leve otimismo, e o teste confirma o resultado.
2. O bootstrap mede só a variabilidade amostral do teste, e não a do treino (um único split 80/20).
3. As colunas de atraso são colineares, então não interpretamos coeficientes individuais da logística.
4. O `DebtRatio` continua misturando razão e valor (quando a renda falta). O modelo distingue os casos pela flag de renda, e a demo pede o valor quando a renda não é informada.
5. A demo não permite informar o código 96/98.
6. Antes das mudanças desta versão, uma análise exploratória do grupo consultou o conjunto de teste para comparar alternativas de modelo. As decisões desta versão se apoiaram na validação cruzada do treino e em critérios de coerência, mas o teste deixou de ser totalmente "não visto" nessa análise.

## 10. Trabalhos futuros

- **Rigor acadêmico, itens restantes:** ablação sem as features derivadas; XGBoost e CatBoost; SMOTE dentro do Pipeline, para demonstrar o efeito na calibração; scorecard tradicional (WoE/IV e escala de pontos); replicação no dataset UCI *Default of Credit Card Clients* (equidade por sexo); *model card*. A busca de hiperparâmetros (14 configurações) e a comparação Platt × isotônica já foram feitas.
- **Produção:** API (FastAPI) e contêiner (Docker), log de auditoria, rastreamento de experimentos (MLflow), monitoramento de estabilidade (PSI/CSI), política de retreino e *champion/challenger*, CI.
- **Decisão:** parâmetros reais do revisor (custo e taxa de acerto medidos), capacidade da equipe de análise e tratamento do incentivo a omitir a renda.
- **Dados:** dataset com data (split temporal), histórico de recusas (reject inference), valor, prazo e taxa reais (LGD/EAD) e bureaus brasileiros.

## 11. Conclusão

Um LightGBM com restrições monotônicas, sem a idade e sem pesos de classe, com corte baseado em custo e explicações por SHAP, alcança AUC 0,864 no teste. A PD é calibrada (Brier 0,048) e contínua, e as decisões são coerentes: mais risco nunca melhora a decisão. Sob premissas de custo ilustrativas, o corte por custo supera o corte padrão e o de F1, mas o ganho sobre aprovar todos é modesto (+13,1%). O LightGBM ordena melhor que a logística por margem pequena, porém estatisticamente significativa (ΔAUC +0,0070; IC 95% de 0,0038 a 0,0102), sem diferença significativa de lucro. A revisão humana fica onde ela tem valor esperado positivo e recebe 3,9% das propostas. Tirar a idade do modelo custou 0,003 de AUC, não teve efeito detectável no lucro e reduziu a diferença de tratamento entre bons pagadores jovens e idosos de 6,6 para 5,5 p.p. A diferença que resta vem de variáveis correlacionadas. O valor do modelo depende fortemente da premissa de custo (de +4,3% a +93,5% sobre aprovar todos), e o corte e a faixa de revisão devem ser recalibrados quando ela muda.

## 12. Referências

- Thomas, Edelman & Crook. *Credit Scoring and Its Applications*. SIAM.
- Siddiqi, N. *Credit Risk Scorecards*. Wiley.
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
- Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers.
- Kaggle (2011). *Give Me Some Credit*.
- Brasil. Lei 13.709/2018 (LGPD).
