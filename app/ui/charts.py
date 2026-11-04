"""Gráficos Plotly com template próprio. Recebem dados já lidos de reports/metrics/*.json; não calculam métricas."""
import plotly.graph_objects as go

from ui.components import br, pct, pd_text
from ui.theme import TOKENS as T

FONT = "Inter, system-ui, Segoe UI, Roboto, Arial, sans-serif"
MODEL_COLOR = {"dummy": "#94A3B8", "logreg": "#D97706", "lgbm": T["navy"]}
MODEL_LABEL = {"dummy": "Dummy (prior)", "logreg": "Regressão logística", "lgbm": "LightGBM"}
CFG = {"displayModeBar": False}


def _base(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    opts = dict(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=13, color=T["text"]), hovermode="closest", separators=",.",
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    opts.update(kw)
    fig.update_layout(**opts)
    fig.update_xaxes(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0", linecolor="#CBD5E1")
    return fig


GAUGE_MAX = 60  # escala visual em %; acima disso o ponteiro satura (o valor real fica no centro)


def gauge(pd_value: float, threshold: float, band: float) -> go.Figure:
    lo, hi = (threshold - band) * 100, (threshold + band) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge", value=min(pd_value * 100, GAUGE_MAX),
        gauge=dict(
            axis=dict(range=[0, GAUGE_MAX], tickvals=[0, 20, 40, GAUGE_MAX], ticktext=["0%", "20%", "40%", f"{GAUGE_MAX}%+"], tickfont=dict(size=12)),
            bar=dict(color=T["navy"], thickness=0.3),
            steps=[dict(range=[0, lo], color="#BBF7D0"), dict(range=[lo, hi], color="#FDE68A"), dict(range=[hi, GAUGE_MAX], color="#FECACA")],
            threshold=dict(line=dict(color=T["navy"], width=4), thickness=0.85, value=threshold * 100),
            bgcolor="#FFFFFF", borderwidth=0,
        ),
    ))
    fig.add_annotation(x=0.5, y=0.12, text=f"<b>{pd_text(pd_value)}</b>", showarrow=False, font=dict(size=30, color=T["navy"]))
    fig.add_annotation(x=0.5, y=-0.16, text=f"corte {pct(threshold)} · revisão ±{br(band * 100, 0)} p.p.", showarrow=False,
                       font=dict(size=12, color=T["muted"]))
    return _base(fig, 185, margin=dict(l=20, r=20, t=5, b=22))


def waterfall(expl, top: int = 8) -> go.Figure:
    """Contribuição de cada variável (log-odds do modelo cru) para o caso atual."""
    import numpy as np

    vals, data, names = np.asarray(expl.values), np.asarray(expl.data), list(expl.feature_names)
    order = np.argsort(-np.abs(vals))
    show, rest = order[:top], order[top:]
    labels = [f"{names[i]} = {br(float(data[i]), 2 if abs(data[i]) < 10 and data[i] % 1 else 0)}" for i in show]
    x = [float(vals[i]) for i in show]
    if len(rest):
        labels.append(f"Outras {len(rest)} variáveis")
        x.append(float(vals[rest].sum()))
    base = float(expl.base_values)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(x) + ["total"],
        y=["Risco médio da base"] + labels + ["Este caso"], x=[base] + x + [0],
        text=[""] + [f"{'+' if v > 0 else ''}{br(v, 2)}" for v in x] + [""], textposition="outside",
        increasing=dict(marker=dict(color=T["reject"])), decreasing=dict(marker=dict(color=T["approve"])),
        totals=dict(marker=dict(color=T["navy"])), connector=dict(line=dict(color="#CBD5E1", width=1)),
        hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="log-odds do modelo cru (positivo = aumenta o risco)")
    return _base(fig, 420, margin=dict(l=10, r=40, t=10, b=40), showlegend=False)


def roc(curves: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94A3B8", dash="dot"), name="Aleatório", hoverinfo="skip"))
    for m in ("dummy", "logreg", "lgbm"):
        c = curves["roc"][m]
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{MODEL_LABEL[m]} (AUC {br(c['auc'], 3)})",
                        line=dict(color=MODEL_COLOR[m], width=3 if m == "lgbm" else 2))
    fig.update_xaxes(title="Taxa de falsos positivos")
    fig.update_yaxes(title="Taxa de verdadeiros positivos")
    return _base(fig)


def pr(curves: dict) -> go.Figure:
    fig = go.Figure()
    for m in ("logreg", "lgbm"):
        c = curves["pr"][m]
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines", name=f"{MODEL_LABEL[m]} (PR-AUC {br(c['pr_auc'], 3)})",
                        line=dict(color=MODEL_COLOR[m], width=3 if m == "lgbm" else 2))
    fig.add_hline(y=curves["pr"]["baseline"], line=dict(color="#94A3B8", dash="dot"),
                  annotation_text=f"base = {pct(curves['pr']['baseline'])}", annotation_position="bottom right")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precisão")
    return _base(fig)


def ks(curves: dict) -> go.Figure:
    k = curves["ks"]
    fig = go.Figure()
    fig.add_scatter(x=k["x"], y=k["cdf_good"], mode="lines", name="Bons pagadores", line=dict(color=T["approve"], width=3))
    fig.add_scatter(x=k["x"], y=k["cdf_bad"], mode="lines", name="Maus pagadores", line=dict(color=T["reject"], width=3))
    fig.add_vline(x=k["x_ks"], line=dict(color=T["navy"], dash="dash"), annotation_text=f"KS = {br(k['ks'], 3)}")
    fig.update_xaxes(title="PD calibrada")
    fig.update_yaxes(title="Distribuição acumulada")
    return _base(fig)


def calibration(curves: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94A3B8", dash="dot"), name="Calibração perfeita", hoverinfo="skip"))
    for key, label, color in (("before", "Antes (probabilidade crua)", T["reject"]), ("after", "Depois (isotônica)", T["navy"])):
        c = curves["calibration"][key]
        fig.add_scatter(x=c["mean_pred"], y=c["frac_pos"], mode="lines+markers", name=label, line=dict(color=color, width=3))
    fig.update_xaxes(title="PD prevista (média do decil)", range=[0, 1])
    fig.update_yaxes(title="Taxa observada de maus", range=[0, 1])
    return _base(fig)


def profit(curves: dict) -> go.Figure:
    p = curves["profit"]
    fig = go.Figure(go.Scatter(x=p["thresholds"], y=p["profit_per_proposal"], mode="lines", line=dict(color=T["navy"], width=3),
                               name="Lucro esperado", hovertemplate="corte %{x:.3f}<br>lucro %{y:.4f}<extra></extra>"))
    style = {"padrao_0.5": ("Corte 0,5", "#64748B"), "f1": ("Corte F1", T["review"]),
             "custo": ("Corte por custo", T["approve"]), "teorico_0.286": ("Teórico 0,2857", T["reject"])}
    for k, (label, color) in style.items():
        m = p["marks"][k]
        fig.add_scatter(x=[m["corte"]], y=[m["lucro"]], mode="markers", name=f"{label}: {br(m['corte'], 3)} → {br(m['lucro'], 4)}",
                        marker=dict(color=color, size=13, line=dict(color="#fff", width=2),
                                    symbol={"padrao_0.5": "square", "f1": "diamond", "custo": "circle", "teorico_0.286": "triangle-up"}[k]))
    fig.update_xaxes(title="Corte (aprova se PD < corte)")
    fig.update_yaxes(title="Lucro esperado por proposta (premissas ilustrativas)", range=[0.3, 0.72])
    return _base(fig, 420)


def importance(imp: dict, labels: dict, top: int = 10) -> go.Figure:
    items = list(imp.items())[:top]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[labels.get(k, k) for k, _ in items], orientation="h", marker_color=T["navy"],
                           text=[br(v, 3) for _, v in items], textposition="outside"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Importância média |SHAP| (log-odds)", range=[0, max(v for _, v in items) * 1.2])
    return _base(fig, 400, showlegend=False, margin=dict(l=10, r=40, t=10, b=40))


def fairness_bars(rows: list[dict]) -> go.Figure:
    g = [f"{r['faixa']}<br>(n={br(r['n'], 0)})" for r in rows]
    fig = go.Figure()
    for key, label, color in (("taxa_aprovacao", "Taxa de aprovação", T["navy"]), ("tpr_bom_aprovado", "Bons pagadores aprovados (TPR)", T["accent"]),
                              ("taxa_mau_real", "Inadimplência real", "#94A3B8")):
        fig.add_bar(x=g, y=[r[key] * 100 for r in rows], name=label, marker_color=color, text=[pct(r[key]) for r in rows], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="%", range=[0, 112])
    return _base(fig, 420)


def di_bars(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[r["faixa"] for r in rows], y=[r["disparate_impact"] for r in rows], marker_color=T["navy"],
                           text=[br(r["disparate_impact"], 3) for r in rows], textposition="outside", name="Disparate impact"))
    fig.add_scatter(x=[rows[0]["faixa"], rows[-1]["faixa"]], y=[0.8, 0.8], mode="lines", name="Limite 0,8 (regra dos 4/5)",
                    line=dict(color=T["reject"], dash="dash", width=3))
    fig.update_yaxes(title="Disparate impact (vs. grupo de maior aprovação)", range=[0, 1.15])
    return _base(fig, 420)
