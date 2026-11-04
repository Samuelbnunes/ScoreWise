| Modelo | AUC | Gini | KS | PR-AUC | Brier (cru) | Brier (calibrado) |
|---|---|---|---|---|---|---|
| Dummy (prior) | 0.5000 | 0.0000 | 0.0000 | 0.0670 | 0.0625 | 0.0625 |
| Regressão logística | 0.8590 | 0.7181 | 0.5655 | 0.3908 | 0.1459 | 0.0494 |
| LightGBM | 0.8648 | 0.7296 | 0.5815 | 0.4037 | 0.1363 | 0.0489 |

Modelo final: **LightGBM** (calibrado). Teste: n=29879, inadimplência 6.70%.

| Regra de corte | Corte | Lucro/proposta | Aprovação | Inadimplência aprovados |
|---|---|---|---|---|
| padrao_0.5 | 0.500 | 0.6384 | 98.18% | 5.71% |
| f1 | 0.210 | 0.6645 | 90.79% | 3.44% |
| custo | 0.270 | 0.6753 | 93.70% | 4.05% |
| teorico_0.286 | 0.286 | 0.6753 | 93.70% | 4.05% |
