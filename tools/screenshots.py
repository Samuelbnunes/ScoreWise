"""Gera screenshots de todas as páginas (1366×768 e 1920×1080) e dos 3 perfis do Simulador.
Ferramenta de desenvolvimento (fora do requirements.txt): pip install playwright && playwright install chromium
Uso: suba o app (streamlit run app/streamlit_app.py --server.port 8601) e rode: python tools/screenshots.py [url]"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8601").rstrip("/")
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
VIEWPORTS = {"1366x768": (1366, 768), "1920x1080": (1920, 1080)}
PAGES = ["inicio", "simulador", "desempenho", "explicabilidade", "equidade", "sobre"]
PROFILES = [("Bom", "bom"), ("Limítrofe", "limitrofe"), ("Arriscado", "arriscado")]


def shot(page, name, w, h):
    """O Streamlit rola dentro de [data-testid=stMain]; amplia o viewport para capturar tudo."""
    total = page.evaluate("document.querySelector('[data-testid=stMain]').scrollHeight")
    page.set_viewport_size({"width": w, "height": max(h, total)})
    page.wait_for_timeout(700)
    page.screenshot(path=str(OUT / name))
    page.set_viewport_size({"width": w, "height": h})


def ready(page):
    page.wait_for_selector(".sw-footer", timeout=90_000)
    page.wait_for_timeout(1800)  # gráficos Plotly e animações


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    for vp_name, (w, h) in VIEWPORTS.items():
        page = browser.new_page(viewport={"width": w, "height": h})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        for name in PAGES:
            page.goto(f"{BASE}/{name}")
            ready(page)
            if name != "simulador":
                shot(page, f"{name}_{vp_name}.png", w, h)
        page.goto(f"{BASE}/simulador")
        ready(page)
        for label, slug in PROFILES:
            page.get_by_role("button", name=re.compile(rf"\b{label}\b")).first.click()
            page.wait_for_timeout(600)
            page.get_by_role("button", name=re.compile("Analisar proposta")).click()
            page.wait_for_selector(".sw-decision", timeout=60_000)
            page.wait_for_selector(".js-plotly-plot", timeout=60_000)
            page.wait_for_timeout(1800)
            if slug == "limitrofe":  # o que a plateia vê sem rolar
                page.evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)")
                page.wait_for_timeout(500)
                page.screenshot(path=str(OUT / f"simulador_limitrofe_{vp_name}_viewport.png"))
            shot(page, f"simulador_{slug}_{vp_name}.png", w, h)
        # fonte realmente aplicada (Inter pode cair no fallback sem internet)
        fam = page.evaluate("getComputedStyle(document.querySelector('.sw-kpi-value, .sw-decision-label, h5')).fontFamily")
        print(vp_name, "| fonte computada:", fam, "| erros JS:", errors or "nenhum")
        page.close()
    browser.close()
print("ok ->", OUT)
