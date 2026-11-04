# Roteiro da apresentação: ScoreWise (12 slides, ~12 min)

Regra: todo número vem de `reports/metrics/`. Figuras em `reports/figures/`. Demo ao vivo no slide 10 (vídeo de backup gravado antes).

| # | Slide | Conteúdo (bullets do slide) | Figura | Fala (≈ tempo) |
|---|---|---|---|---|
| 1 | **ScoreWise: problema e proposta** | Aprovar crédito tem custo assimétrico. Saída: PD calibrada + decisão (aprovar / revisar / recusar) + 3 motivos | — | Apresentar o problema e a proposta em 1 frase (40 s) |
| 2 | **Dados e definição de inadimplência** | *Give Me Some Credit*: 149.391 propostas, 6,70% de maus. Mau = atraso ≥ 90 dias em 2 anos. Não há data, valor, sexo ou região | `eda_alvo.png` | Classe rara: acurácia seria enganosa (50 s) |
| 3 | **3 achados da EDA** | (1) Idade: 11,9% de maus (< 30) contra 3,1% (60+). (2) Um atraso 90+ leva o risco de 4,6% para 33,7%. (3) Utilização do limite: risco salta nos 2 últimos decis (16,4% e 23,5%) | `eda_idade.png`, `eda_atrasos90.png`, `eda_utilizacao.png` | Um achado por figura (60 s) |
| 4 | **Pipeline sem vazamento** | Split 80/20 estratificado antes de tudo. Mediana, clip e escala dentro do Pipeline. Teste usado uma vez. Códigos 96/98 viram flag (taxa de mau 54,6%) | `eda_anomalias.png` | Mostrar o desenho: limpeza estática, split, Pipeline, CV, calibração, teste (60 s) |
| 5 | **Modelos e tabela comparativa** | Teste: Logística AUC 0,859 / KS 0,566; **LightGBM AUC 0,865 / KS 0,582 / Gini 0,730**. CV: 0,8634 ± 0,0025. Bootstrap pareado (1.000×, seed 42): ΔAUC **+0,0058 [0,0031; 0,0090]**, IC não contém 0: **significativa, porém pequena**; vence nos 5 folds | `roc.png` | Ser honesto: a logística chega perto (50 s) |
| 6 | **Calibração: por que importa** | `class_weight` infla a PD. Isotônica sobre OOF. Brier 0,136 → 0,049 | `calibration.png` | "PD de 10% precisa significar 10% de maus" (50 s) |
| 7 | **Corte por custo** | Matriz +1 / −5 / −1 / 0 (ilustrativa). Teórico 0,2857; empírico 0,270. Lucro/proposta: 0,5 → 0,638; F1 → 0,665; **custo → 0,675**; aprovar todos → 0,598. **Sensibilidade à perda por mau (−3 / −5 / −10):** corte ótimo 0,345 / 0,270 / 0,170; ganho sobre aprovar todos +3,4% / +12,9% / +93,6% | `profit_curve.png` | Teoria e prática batem: argumento de calibração; fechar com a sensibilidade: o valor depende da premissa de custo (80 s) |
| 8 | **SHAP global e reason codes** | Top: utilização do limite, total de atrasos, idade. 100% das 1.883 recusas com 3 motivos em português (`idade` não é exibida como motivo: ADR-13) | `shap_summary.png`, `shap_waterfall_arriscado.png` | Um caso recusado, motivo a motivo (70 s) |
| 9 | **Equidade e LGPD** | DI mínimo 0,890 (> 0,8). Mas bons pagadores < 30 anos: 92,4% aprovados contra 98,7% (60+): −6,3 p.p. Faixa de revisão humana ±0,05 (LGPD art. 20: conferir norma vigente). **Sem `idade` (Q5):** AUC −0,0036, lucro igual (IC contém 0), DI 0,890 → 0,902, mas o gap de TPR só cai de 6,3 para 5,3 p.p. (IC 4,0–6,6): o efeito persiste via variáveis correlacionadas | `fairness_age.png` | Destacar a diferença além do risco e mostrar que retirar a idade não resolve (80 s) |
| 10 | **Demo ao vivo** | Perfis: Bom → ✅ APROVAR (PD exibida: < 1%). Limítrofe → 🟡 REVISAR (25,4%). Arriscado → ❌ RECUSAR (68,8%). Mostrar também a aba Explicabilidade | app | Ordem: Bom, Limítrofe, Arriscado; "Analisar" uma vez antes para aquecer (120 s) |
| 11 | **Limitações e roadmap** | Sem data (split não temporal), sem recusados, custos assumidos, equidade só por idade, dados de 2011, protótipo. F2 (restante): ablação sem features derivadas, XGBoost/Optuna, scorecard, UCI (sexo). F3: API, Docker, MLflow, PSI/CSI | — | Limitações honestas valem pontos (50 s) |
| 12 | **Conclusão** | AUC 0,865; PD calibrada; corte por custo +5,8% sobre 0,5; ganho de +12,9% sobre aprovar todos (de +3,4% a +93,6% conforme a perda assumida); ponto de atenção ético: idade | — | 3 frases (30 s) |

## Checklist de ensaio

- [ ] `python src\train.py` e `streamlit run app\streamlit_app.py` abertos antes de começar; clicar em Analisar uma vez (aquece o cache, ~0,6 s na primeira)
- [ ] Vídeo de backup da demo (1–2 min) gravado e testado
- [ ] Ensaio cronometrado (alvo: 10 a 12 min)
- [ ] Perguntas prováveis: *por que o corte não é 0,5?* (slide 7); *por que calibrar?* (slide 6); *o LightGBM é significativamente melhor que a logística?* (sim: IC 95% do ΔAUC [0,0031; 0,0090], mas a margem é pequena); *e se retirar a idade?* (slide 9); *idade pode ser usada?* (slide 9); *os custos são reais?* (não: premissas ilustrativas)
