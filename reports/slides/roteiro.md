# Roteiro da apresentação: ScoreWise (12 slides, ~12 min)

Regra: todo número vem de `reports/metrics/`. Figuras em `reports/figures/`. Demo ao vivo no slide 10 (vídeo de backup gravado antes). Modelo v0.2.0.

| # | Slide | Conteúdo (bullets do slide) | Figura | Fala (≈ tempo) |
|---|---|---|---|---|
| 1 | **ScoreWise: problema e proposta** | Aprovar crédito tem custo assimétrico. Saída: PD calibrada + decisão (aprovar / revisar / recusar) + até 3 motivos | — | Apresentar o problema e a proposta em 1 frase (40 s) |
| 2 | **Dados e definição de inadimplência** | *Give Me Some Credit*: 149.352 propostas, 6,70% de maus. Mau = atraso ≥ 90 dias em 2 anos. Não há data, valor, sexo ou região | `eda_alvo.png` | Classe rara: acurácia seria enganosa (50 s) |
| 3 | **Achados da EDA** | (1) Idade: 11,9% de maus (< 30) contra 3,1% (60+). (2) Um atraso 90+ leva o risco de 4,6% para 33,7%. (3) Utilização do limite: risco salta nos 2 últimos decis (16,4% e 23,5%); quem não usa rotativo tem 2,94% de maus contra 1,28% de quem usa até 1%. (4) Renda 0 = dívida em valor, como renda nula | `eda_idade.png`, `eda_atrasos90.png`, `eda_utilizacao.png` | Um achado por figura; o 3 e o 4 explicam decisões do modelo (60 s) |
| 4 | **Pipeline sem vazamento** | Split 80/20 estratificado antes de tudo. Mediana, clip e escala dentro do Pipeline. Teste usado uma vez. Códigos 96/98 viram flag (taxa de mau 54,6%); renda 0 = não informada | `eda_anomalias.png` | Mostrar o desenho: limpeza estática, split, Pipeline, CV, calibração, teste (60 s) |
| 5 | **Modelos e tabela comparativa** | Mesmas 15 variáveis, sem peso de classe. Teste: logística AUC 0,857 / KS 0,553; **LightGBM AUC 0,864 / KS 0,567 / Gini 0,729**. CV: 0,8597 ± 0,0067. Bootstrap pareado: ΔAUC **+0,0070 [0,0038; 0,0102]** (significativa, porém pequena; vence nos 5 folds); Δlucro **não** significativo | `roc.png` | Ser honesto: a logística chega perto, e o lucro empata (50 s) |
| 6 | **Coerência e calibração** | Restrições monotônicas: mais atraso, uso do limite ou dívida nunca reduz o risco; mais renda nunca o aumenta. Varreduras nos 3 perfis: 0 reversões de decisão. Sem `class_weight`, a PD já sai calibrada (Brier 0,048); Platt ajusta o resto; PD contínua | `calibration.png` | "Uma PD de 10% precisa significar 10% de maus, e mais dívida nunca pode melhorar a decisão" (60 s) |
| 7 | **Corte por custo e revisão humana** | Matriz +1 / −5 / −1 / 0 (ilustrativa). Teórico 0,2857; empírico 0,260. Lucro/proposta: 0,5 → 0,646; F1 → 0,673; **custo → 0,676**; aprovar todos → 0,598. Revisão humana onde vale mais que a decisão automática: PD de 20,7% a 34,1% (3,9% das propostas, 28% de maus). **Sensibilidade à perda por mau (−3 / −5 / −10):** corte ótimo 0,400 / 0,260 / 0,160; ganho sobre aprovar todos +4,3% / +13,1% / +93,5% | `profit_curve.png` | A faixa de revisão tem justificativa econômica; fechar com a sensibilidade: o valor depende da premissa de custo (80 s) |
| 8 | **SHAP global e motivos** | Top: utilização do limite, total de atrasos, razão dívida/renda. Até 3 motivos distintos: as variáveis de atraso viram um único motivo (o atraso mais grave). 1.279 das 1.304 recusas com 3 motivos; todas com pelo menos 1 | `shap_summary.png`, `shap_waterfall_arriscado.png` | Um caso recusado, motivo a motivo (70 s) |
| 9 | **Equidade e LGPD** | A idade **não é usada** pelo modelo (só audita). DI mínimo 0,900 (> 0,8). Bons pagadores < 30 anos: 92,7% aprovados contra 98,3% (60+): −5,5 p.p. **Com a idade no modelo (contrafactual):** AUC +0,003, lucro igual (IC contém 0), diferença de TPR 6,6 p.p.: tirar a idade reduz a disparidade em 1,1 p.p. (significativo), mas não a elimina (variáveis correlacionadas). Revisão humana (LGPD art. 20; conferir a norma vigente) | `fairness_age.png` | Destacar a diferença além do risco e por que a idade saiu (80 s) |
| 10 | **Demo ao vivo** | Perfis: Bom → ✅ APROVAR (PD < 1%). Limítrofe → 🟡 REVISAR (25,7%). Arriscado → ❌ RECUSAR (61,4%). Mostrar também: renda 0 é tratada como não informada (aviso na tela) e a página Equidade | app | Ordem: Bom, Limítrofe, Arriscado; "Analisar" uma vez antes para aquecer (120 s) |
| 11 | **Limitações e roadmap** | Sem data (split não temporal), sem recusados, custos e revisor assumidos, equidade só por idade, dados de 2011, renda não informada com PD um pouco menor (incentivo a omitir), protótipo. Futuro: XGBoost/CatBoost, scorecard, UCI (sexo), revisor real. Produção: API, Docker, MLflow, PSI/CSI | — | Limitações honestas valem pontos (50 s) |
| 12 | **Conclusão** | AUC 0,864; PD calibrada e contínua; decisões coerentes; corte por custo +4,8% sobre 0,5 e +13,1% sobre aprovar todos (de +4,3% a +93,5% conforme a perda assumida); idade fora do modelo, com disparidade menor mas não nula | — | 3 frases (30 s) |

## Checklist de ensaio

- [ ] `streamlit run app\streamlit_app.py` aberto antes de começar; clicar em Analisar uma vez (aquece o cache)
- [ ] Vídeo de backup da demo (1–2 min) gravado e testado
- [ ] Ensaio cronometrado (alvo: 10 a 12 min)
- [ ] Perguntas prováveis:
  - *Por que o corte não é 0,5?* (slide 7)
  - *Por que treinar sem `class_weight`?* (slide 6: mesma AUC e PD já calibrada; o peso de classe só obrigava a recalibrar)
  - *O LightGBM é significativamente melhor que a logística?* (em AUC sim, IC 95% do ΔAUC [0,0038; 0,0102], mas a margem é pequena e o lucro empata)
  - *Por que tirar a idade?* (slide 9: atributo sensível; sem ganho detectável de lucro; reduz a disparidade)
  - *O que é a faixa de revisão?* (slide 7: onde revisar vale mais que decidir; depende de premissas do revisor)
  - *Por que restrições monotônicas?* (slide 6: coerência das decisões e das explicações; custo de ~0,002 de AUC)
  - *Os custos são reais?* (não: premissas ilustrativas)
