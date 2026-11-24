# Equidade por faixa etária (teste, LightGBM v0.2.0, corte 0,260)

A idade **não é variável do modelo** desde a v0.2.0. Ela é usada aqui só para auditar o resultado e é o único atributo sensível do dataset. Fonte: `reports/metrics/fairness.json` e `reports/metrics/stats.json`.

**Corte simples (aprova se PD < 0,260):**

| Faixa | n | Mau real | Aprovação | DI | TPR (bom aprovado) | Dif. TPR vs 60+ |
|---|---|---|---|---|---|---|
| <30 | 1.673 | 12,67% | 87,51% | 0,900 | 92,74% | −5,54 p.p. |
| 30–44 | 7.786 | 9,21% | 91,18% | 0,938 | 94,86% | −3,42 p.p. |
| 45–59 | 10.756 | 7,28% | 93,29% | 0,960 | 96,14% | −2,14 p.p. |
| 60+ | 9.656 | 3,00% | 97,18% | 1,000 | 98,28% | 0 |

**Decisão em 3 faixas do produto** (aprovação automática se PD < 20,7%; revisão humana de 20,7% a 34,1%; recusa acima):

| Faixa | Aprovação automática | Revisão humana | Recusa | Bons aprovados automaticamente | DI (automático) |
|---|---|---|---|---|---|
| <30 | 83,80% | 8,25% | 7,95% | 89,73% | 0,872 |
| 30–44 | 88,67% | 5,21% | 6,11% | 92,81% | 0,923 |
| 45–59 | 91,35% | 3,85% | 4,80% | 94,63% | 0,951 |
| 60+ | 96,10% | 2,05% | 1,85% | 97,36% | 1,000 |

**Interpretação.**
- **Disparate impact:** o mínimo é 0,900 no corte simples e 0,872 na aprovação automática (faixa <30 contra 60+). Os dois ficam acima do limite de 0,8 da regra dos 4/5, então nenhuma faixa dispara o alerta. Esse limite é uma heurística de origem estadunidense, e não uma norma brasileira.
- **Diferença explicada pelo risco:** a diferença de aprovação é em grande parte explicada pelo risco, porque a inadimplência observada em <30 (12,67%) é 4,2 vezes a de 60+ (3,00%).
- **Diferença além do risco:** mesmo assim, a *equal opportunity* mostra uma disparidade além do risco médio do grupo. Entre os **bons pagadores**, 92,74% dos menores de 30 anos são aprovados, contra 98,28% dos 60+: uma diferença de 5,5 pontos percentuais (IC 95% por bootstrap: 4,2 a 6,9). Um bom pagador jovem é recusado cerca de 4 vezes mais (7,26% contra 1,72%) que um bom pagador idoso.
- **Origem da diferença:** como a idade não está no modelo, a diferença vem de variáveis correlacionadas com ela (renda, dependentes, linhas de crédito, histórico de atrasos). Na decisão em 3 faixas, os menores de 30 também vão mais para a revisão humana (8,25% contra 2,05%).

**E se a idade estivesse no modelo?** Treinamos o contrafactual, com a mesma configuração e o mesmo protocolo. Com a idade, a diferença de TPR seria de 6,62 p.p. (IC 5,3 a 8,0), contra 5,54 p.p. do modelo final. A redução de 1,08 p.p. (IC 0,44 a 1,68) é estatisticamente significativa. A AUC seria 0,0029 maior e o lucro não muda de forma detectável. Tirar a idade reduz a disparidade, mas não a elimina.

**Limites da análise.** Os dados não permitem analisar sexo ou região (L4). A faixa <30 tem apenas 1.673 propostas, então as taxas dela têm mais incerteza.
