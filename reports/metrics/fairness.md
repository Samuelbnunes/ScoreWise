# Equidade por faixa etária (teste, LightGBM calibrado, corte 0,270)

| Faixa | n | Mau real | Aprovação | DI | TPR (bom aprovado) | Dif. TPR vs 60+ |
|---|---|---|---|---|---|---|
| <30 | 1.781 | 11,90% | 86,81% | 0,890 | 92,35% | −6,31 p.p. |
| 30–44 | 7.869 | 9,07% | 90,99% | 0,933 | 94,69% | −3,98 p.p. |
| 45–59 | 10.698 | 7,16% | 93,43% | 0,958 | 96,05% | −2,61 p.p. |
| 60+ | 9.531 | 3,25% | 97,52% | 1,000 | 98,67% | 0 |

**Interpretação.** O *disparate impact* mínimo é 0,890 (faixa <30 contra 60+), acima do limite de 0,8 da regra dos 4/5, então nenhuma faixa dispara o alerta. Esse limite é uma heurística de origem estadunidense, e não uma norma brasileira. A diferença de aprovação é em grande parte explicada pelo risco: a inadimplência observada em <30 (11,9%) é 3,7 vezes a de 60+ (3,25%). Mesmo assim, a *equal opportunity* mostra uma disparidade além do risco médio do grupo: entre os **bons pagadores**, 92,35% dos menores de 30 anos são aprovados contra 98,67% dos 60+, uma diferença de 6,3 pontos percentuais. Isto é, um bom pagador jovem é recusado cerca de 5 vezes mais (7,65% contra 1,33%) do que um bom pagador idoso. O efeito vem de `idade` ser a 3ª variável mais importante do modelo (|SHAP| médio 0,30) e de o modelo não distinguir o bom pagador jovem dos maus pagadores jovens com a mesma precisão. A diferença de risco real não justifica esse erro adicional. A ablação sem `idade` (Q5) e a sensibilidade de custos ficam para a F2. Os dados não permitem analisar sexo ou região (L4), e a faixa <30 tem apenas 1.781 propostas, então as taxas dela têm mais incerteza.
