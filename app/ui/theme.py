"""Tokens visuais, CSS global e logo. Tema claro fixo e alto contraste (apresentação em projetor)."""

TOKENS = {
    "navy": "#0B1F3A",
    "accent": "#10B981",
    "bg": "#F7F9FC",
    "surface": "#FFFFFF",
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "approve": "#16A34A",
    "review": "#D97706",
    "reject": "#DC2626",
}
DECISION_COLOR = {"APROVAR": TOKENS["approve"], "REVISAR": TOKENS["review"], "RECUSAR": TOKENS["reject"]}
DECISION_TINT = {"APROVAR": "#ECFDF3", "REVISAR": "#FFF7E6", "RECUSAR": "#FEF0F0"}

TAGLINE = "Decisões de crédito explicáveis"

# Monograma original: escudo azul-marinho com "SW" e uma seta ascendente esmeralda.
LOGO_SVG = (
    '<svg width="38" height="38" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ScoreWise">'
    '<path d="M24 3 42 9v14c0 11-7.5 19.5-18 22C13.5 42.500 6 34 6 23V9z" fill="#0B1F3A"/>'
    '<path d="M13 31l7-8 5 5 10-12" fill="none" stroke="#10B981" stroke-width="3.400" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M29 16h6v6" fill="none" stroke="#10B981" stroke-width="3.400" stroke-linecap="round" stroke-linejoin="round"/>'
    '<text x="24" y="40" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-weight="800" font-size="9" fill="#FFFFFF">SW</text>'
    "</svg>"
)

_T = TOKENS
CSS = f"""
/* Inter servida localmente (app/static/fonts); o Google Fonts é só o 2º src, usado se o arquivo local falhar */
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 400; font-display: swap; src: url('/app/static/fonts/inter-latin-400-normal.woff2') format('woff2'), url('https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2') format('woff2'); }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 600; font-display: swap; src: url('/app/static/fonts/inter-latin-600-normal.woff2') format('woff2'), url('https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2') format('woff2'); }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 700; font-display: swap; src: url('/app/static/fonts/inter-latin-700-normal.woff2') format('woff2'), url('https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2') format('woff2'); }}
@font-face {{ font-family: 'Inter'; font-style: normal; font-weight: 800; font-display: swap; src: url('/app/static/fonts/inter-latin-800-normal.woff2') format('woff2'), url('https://fonts.gstatic.com/s/inter/v20/UcC73FwrK3iLTeHuS_nVMrMxCp50SjIa1ZL7.woff2') format('woff2'); }}
:root {{ --sw-navy:{_T['navy']}; --sw-accent:{_T['accent']}; --sw-bg:{_T['bg']}; --sw-surface:{_T['surface']};
        --sw-text:{_T['text']}; --sw-muted:{_T['muted']}; --sw-border:{_T['border']}; }}
/* Fonte: tudo, exceto a fonte de ícones do Streamlit */
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="icon"]) {{ font-family: Inter, system-ui, "Segoe UI", Roboto, Arial, sans-serif !important; }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: var(--sw-text); }}
.stApp {{ background: var(--sw-bg); font-variant-numeric: tabular-nums; }}
/* Esconde menu, rodapé e header padrão do Streamlit (a navegação é própria) */
#MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton, [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding-top: .5rem !important; padding-bottom: 1rem !important; max-width: 1360px; }}
h1, h2, h3 {{ color: var(--sw-navy); letter-spacing: -0.01em; }}

/* Cards (containers com borda) */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--sw-surface); border: 1px solid var(--sw-border) !important;
  border-radius: 12px !important; box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.04); }}
.st-key-sw-nav [data-testid="stVerticalBlockBorderWrapper"] {{ box-shadow: none; border: none !important; background: transparent; }}

/* Navegação superior */
.st-key-sw-nav {{ border-bottom: 1px solid var(--sw-border); padding-bottom: .3rem; margin-bottom: .2rem; }}
.st-key-sw-nav [data-testid="stPageLink"] a {{ border-radius: 8px; padding: .45rem .8rem; font-weight: 600; color: var(--sw-muted); text-decoration: none; }}
.st-key-sw-nav [data-testid="stPageLink"] a:hover {{ background: #EEF2F7; color: var(--sw-navy); }}
.sw-brand {{ display: flex; align-items: center; gap: .6rem; }}
.sw-brand-name {{ font-weight: 800; font-size: 1.25rem; color: var(--sw-navy); line-height: 1.1; }}
.sw-brand-tag {{ font-size: .72rem; color: var(--sw-muted); }}

/* Botões */
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; border: 1px solid var(--sw-border); background: #fff; color: var(--sw-navy); }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--sw-accent); color: var(--sw-navy); }}
.stButton > button[kind="primary"] {{ background: var(--sw-navy); color: #fff; border-color: var(--sw-navy); }}
.stButton > button[kind="primary"]:hover {{ background: #12305a; color: #fff; }}
.stButton > button[kind="primary"] p {{ color: #fff; }}

/* Componentes */
.sw-kpi {{ background: var(--sw-surface); border: 1px solid var(--sw-border); border-radius: 12px; padding: 1rem 1.1rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.04); height: 100%; }}
.sw-kpi-label {{ font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; color: var(--sw-muted); }}
.sw-kpi-value {{ font-size: 2rem; font-weight: 800; color: var(--sw-navy); line-height: 1.15; margin: .15rem 0; }}
.sw-kpi-sub {{ font-size: .8rem; color: var(--sw-muted); }}
.sw-section {{ margin: .8rem 0 .5rem; }}
.sw-section h2 {{ margin: 0; font-size: 1.35rem; }}
.sw-section p {{ margin: .15rem 0 0; color: var(--sw-muted); font-size: .92rem; }}
.sw-hero {{ background: linear-gradient(135deg, #0B1F3A 0%, #12305a 60%, #0f4a56 100%); border-radius: 16px; padding: 2.4rem 2.2rem; color: #fff; }}
.sw-hero h1 {{ color: #fff; font-size: 2.5rem; font-weight: 800; margin: 0 0 .4rem; }}
.sw-hero p {{ color: #CBD5E1; font-size: 1.1rem; margin: 0; max-width: 46rem; }}
.sw-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; align-items: stretch; }}
.sw-gcard {{ background: var(--sw-surface); border: 1px solid var(--sw-border); border-radius: 12px; padding: 1.1rem 1.2rem; box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 4px 14px rgba(15,23,42,.04); }}
.sw-step-n {{ display: inline-flex; width: 2rem; height: 2rem; border-radius: 50%; background: var(--sw-accent); color: #04271c;
  font-weight: 800; align-items: center; justify-content: center; margin-bottom: .5rem; }}
.sw-step h4 {{ margin: 0 0 .25rem; color: var(--sw-navy); }}
.sw-step p {{ margin: 0; color: var(--sw-muted); font-size: .92rem; }}
.sw-decision {{ border-radius: 14px; padding: 1.1rem 1.3rem; display: flex; align-items: center; gap: 1rem; border: 2px solid;
  animation: sw-fade-in .45s ease-out both; }}
.sw-decision-label {{ font-size: 2rem; font-weight: 800; line-height: 1.1; }}
.sw-decision-sub {{ font-size: .95rem; color: var(--sw-text); }}
.sw-chip {{ display: flex; align-items: center; gap: .7rem; padding: .55rem .8rem; border: 1px solid var(--sw-border); border-radius: 10px;
  background: #fff; margin-bottom: .45rem; animation: sw-fade-in .45s ease-out both; }}
.sw-chip-rank {{ width: 1.5rem; height: 1.5rem; border-radius: 50%; background: var(--sw-navy); color: #fff; font-size: .8rem; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center; flex: none; }}
.sw-chip-text {{ flex: 1; font-weight: 600; font-size: .92rem; }}
.sw-chip-bar {{ width: 90px; height: 8px; background: #EEF2F7; border-radius: 4px; overflow: hidden; flex: none; }}
.sw-chip-bar > div {{ height: 100%; border-radius: 4px; }}
.sw-chip-pct {{ width: 5.2rem; text-align: right; font-size: .95rem; font-weight: 700; color: var(--sw-text); flex: none; line-height: 1.05; }}
.sw-chip-pct small {{ display: block; font-size: .62rem; font-weight: 500; color: var(--sw-muted); }}
.sw-empty {{ border: 2px dashed var(--sw-border); border-radius: 14px; padding: 2.2rem 1rem; text-align: center; color: var(--sw-muted); background: #fff; }}
.sw-footer {{ margin-top: 2rem; padding-top: .8rem; border-top: 1px solid var(--sw-border); font-size: .76rem; color: var(--sw-muted); text-align: center; }}
.sw-pill {{ display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 700; background: #E6F4EF; color: #065F46; }}
@keyframes sw-fade-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
"""
