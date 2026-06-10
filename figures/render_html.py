"""Render an HTML file's #wrap element to a crisp PNG via headless Chromium.
Usage: python figures/render_html.py figures/architectures.html figures/model_architectures.png"""
import sys, pathlib
from playwright.sync_api import sync_playwright

html, out = sys.argv[1], sys.argv[2]
url = pathlib.Path(html).resolve().as_uri()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=2)
    pg.set_viewport_size({"width": 1160, "height": 960})
    pg.goto(url); pg.wait_for_timeout(400)
    el = pg.query_selector("#wrap")
    el.screenshot(path=out)
    b.close()
print("wrote", out)
