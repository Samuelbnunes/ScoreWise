"""Confirma que o app funciona sem internet: aborta toda requisição fora de localhost e verifica fonte, decisão e gauge.
Uso: com o app em http://localhost:8601 -> python tools/offline_check.py"""
import re
import sys
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8601").rstrip("/")
blocked, font_requests = [], []

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1366, "height": 768})

    def handler(route):
        host = urlparse(route.request.url).hostname
        if host in ("localhost", "127.0.0.1"):
            route.continue_()
        else:
            blocked.append(route.request.url)
            route.abort()

    ctx.route("**/*", handler)
    page = ctx.new_page()
    page.on("response", lambda r: font_requests.append((r.status, r.url)) if ".woff2" in r.url else None)
    page.goto(f"{BASE}/simulador")
    page.wait_for_selector(".sw-footer", timeout=90_000)
    page.get_by_role("button", name=re.compile(r"\bLimítrofe\b")).first.click()
    page.wait_for_timeout(500)
    page.get_by_role("button", name=re.compile("Analisar proposta")).click()
    page.wait_for_selector(".sw-decision", timeout=60_000)
    page.wait_for_selector(".js-plotly-plot", timeout=60_000)
    page.wait_for_timeout(1500)

    decisao = page.inner_text(".sw-decision-label")
    families = page.evaluate("getComputedStyle(document.querySelector('.sw-decision-label')).fontFamily")
    ok_600 = page.evaluate("document.fonts.check('600 16px Inter')")
    ok_700 = page.evaluate("document.fonts.check('700 16px Inter')")
    loaded = page.evaluate("[...document.fonts].filter(f => f.family.includes('Inter') && f.status === 'loaded').map(f => f.weight)")
    chips = page.locator(".sw-chip").count()
    browser.close()

print("decisão exibida:", decisao, "| chips:", chips)
print("fonte computada:", families)
print("document.fonts.check 600/700:", ok_600, ok_700, "| pesos Inter carregados:", sorted(set(loaded)))
print("woff2 servidos:", [(s, u.split("/")[-1]) for s, u in font_requests])
print("requisições externas bloqueadas:", blocked or "nenhuma")
external_fonts = [u for u in blocked if "font" in u]
sys.exit(0 if decisao and chips == 3 and ok_600 and ok_700 and not external_fonts else 1)
