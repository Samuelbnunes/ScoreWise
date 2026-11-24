| Modelo | AUC | Gini | KS | PR-AUC | Brier (cru) | Brier (calibrado) |
|---|---|---|---|---|---|---|
| Dummy (prior) | 0.5000 | 0.0000 | 0.0000 | 0.0670 | 0.0625 | 0.0625 |
| Regressão logística | 0.8574 | 0.7148 | 0.5531 | 0.3960 | 0.0497 | 0.0497 |
| LightGBM | 0.8644 | 0.7288 | 0.5673 | 0.4199 | 0.0483 | 0.0483 |

Modelo final: **LightGBM** (calibrado). Teste: n=29871, inadimplência 6.70%.

| Regra de corte | Corte | Lucro/proposta | Aprovação | Inadimplência aprovados |
|---|---|---|---|---|
| padrao_0.5 | 0.500 | 0.6457 | 98.14% | 5.59% |
| f1 | 0.210 | 0.6729 | 91.92% | 3.61% |
| custo | 0.260 | 0.6764 | 93.67% | 4.03% |
| teorico_0.286 | 0.286 | 0.6764 | 94.40% | 4.22% |

Decisão em 3 faixas (produto): REVISAR se 0.207 ≤ PD ≤ 0.341. Lucro esperado por proposta com o revisor ilustrativo: 0.6813.

| Faixa | Propostas | % | Taxa de mau |
|---|---|---|---|
| APROVAR | 27411 | 91.76% | 3.59% |
| REVISAR | 1156 | 3.87% | 28.29% |
| RECUSAR | 1304 | 4.37% | 53.07% |
