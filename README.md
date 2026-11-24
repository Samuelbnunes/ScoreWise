# ScoreWise

Modelo explicável de aprovação de crédito (projeto acadêmico). Para cada proposta, retorna a probabilidade de inadimplência (PD) calibrada, a decisão (aprovar / revisar / recusar) e até 3 motivos distintos em português. Modelo v0.2.0: LightGBM com restrições monotônicas, sem a idade como variável.

Dados: *Give Me Some Credit* (Kaggle). Ambiente: Windows, **Python 3.13** (a `.venv` do projeto usa o 3.13; as versões de todas as bibliotecas estão fixadas no `requirements.txt`).

> Protótipo acadêmico. As premissas de custo são ilustrativas. Não usar para decisões reais de crédito.

## Como rodar (PowerShell, na pasta do projeto)

```powershell
# 1. Instalar
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# 2. Dados: baixe cs-training.csv em https://www.kaggle.com/c/GiveMeSomeCredit/data
#    (aceite as regras da competição) e coloque em data\raw\cs-training.csv

# 3. Treinar (gera models\model.joblib, ~30 s) e gerar métricas e figuras
python src\train.py; python src\evaluate.py; python src\explain.py; python src\fairness.py; python src\stats.py

# 4. Demo
streamlit run app\streamlit_app.py
```

## Demo

Seis páginas (Início, Simulador, Desempenho, Explicabilidade, Equidade, Sobre). No **Simulador**, use os perfis rápidos (Bom, Limítrofe, Arriscado) e clique em *Analisar proposta*.

![Simulador](reports/screenshots/simulador_limitrofe_1366x768_viewport.png)

Mais capturas (1366×768 e 1920×1080) em `reports/screenshots/`. Para regenerá-las, com o app rodando na porta 8601: `python tools/screenshots.py`. Para conferir que a demo funciona sem internet: `python tools/offline_check.py`. As duas ferramentas usam o Playwright (`pip install playwright` e `playwright install chromium`; fora do `requirements.txt`, já instalado na `.venv` desta máquina).

Testes: `pytest` (rode depois do treino; `test_predict.py` e `test_app.py` precisam do `model.joblib`; sem o `cs-training.csv`, os testes que dependem dele são pulados).

## O que cada comando gera

| Comando | Saída |
|---|---|
| `python src\train.py` | `models\model.joblib`, `reports\metrics\cv_results.json` |
| `python src\evaluate.py` | `reports\metrics\test_results.{json,md}`, ROC, PR, KS, calibração, curva de lucro, matriz de confusão |
| `python src\explain.py` | SHAP global, waterfalls dos 3 perfis, `explain.json` |
| `python src\fairness.py` | `fairness.json` (corte simples e decisão em 3 faixas por faixa etária), `fairness_age.png`. O texto de `reports\metrics\fairness.md` é escrito à mão a partir do JSON |
| `python src\stats.py` | `stats.json`: IC por bootstrap, contrafactual com idade, sensibilidade de custos e da faixa de revisão (~2 min) |

O treino é reprodutível: `random_state=42` e duas execuções geram o mesmo `cv_results.json`. O teste (`evaluate.py`) só é usado para a avaliação final.

## Estrutura

```
.docs/architecture.md   especificação, decisões (ADRs) e backlog
data/raw/               cs-training.csv (fora do git)
notebooks/01_eda.ipynb  EDA (figuras em reports/figures/eda_*.png)
src/                    config, data, features, calibration, train, evaluate, explain, fairness, stats, predict
app/                    demo (streamlit_app.py, pages/, ui/)
tools/                  screenshots.py (capturas da demo) e offline_check.py (demo sem internet); Playwright, dev
models/                 model.joblib (gerado, fora do git)
reports/                figures/, metrics/, relatorio.md, slides/roteiro.md
tests/                  test_data, test_features, test_predict, test_review, test_app
```

## Documentação

- Relatório: [reports/relatorio.md](reports/relatorio.md)
- Roteiro da apresentação: [reports/slides/roteiro.md](reports/slides/roteiro.md)
- Arquitetura e backlog: [.docs/architecture.md](.docs/architecture.md)
- Plano de melhorias da v0.2.0 e log das alterações: [PLANO_MELHORIAS.md](PLANO_MELHORIAS.md), [LOG_MELHORIAS.md](LOG_MELHORIAS.md)
