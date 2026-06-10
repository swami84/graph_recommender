#!/usr/bin/env python3
"""
Google Maps Reviews Scraper (camoufox + real Google cookies)
Scrapes reviews for restaurants identified by Google Place IDs.

Setup (one-time)
----------------
Close Chrome completely, then run once in the notebook:
    await scrape_reviews.setup_session()
This opens your real Chrome profile, visits Google Maps, and saves cookies
to data/.maps_cookies.json. Subsequent scraping runs reuse those cookies.

Notebook usage
--------------
    import importlib, scrape_reviews
    importlib.reload(scrape_reviews)
    reviews = await scrape_reviews.scrape_reviews("ChIJ...", target=100, headless=False)

CLI usage
---------
    python scrape_reviews.py --setup                          # one-time cookie extraction
    python scrape_reviews.py --place-id ChIJM3elBidawokRJKPC55N6pQI
    python scrape_reviews.py --limit 3 --headless
"""

import argparse
import asyncio
import datetime
import json
import logging
import re
import time
from pathlib import Path

from camoufox.async_api import AsyncCamoufox
from playwright.async_api import async_playwright, Page, Locator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


class NoReviewsTabError(Exception):
    """Reviews tab not found — business likely has no reviews or page is stripped."""

class BotDetectionError(Exception):
    """Scrollable reviews panel missing after clicking tab — likely bot detection."""

# ── Paths ─────────────────────────────────────────────────────────────────────
RESTAURANTS_DIR   = Path("data/restaurants")
REVIEWS_DIR       = Path("data/reviews")
COOKIES_FILE      = Path("data/.maps_cookies.json")   # saved Google cookies
REAL_CHROME_PROFILE = Path.home() / ".config/google-chrome"

REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# ── CSS Selectors ─────────────────────────────────────────────────────────────
SEL_REVIEWS_TAB  = "div.Gpq6kf.NlVald"
SEL_REVIEW_ITEM  = "div.jftiEf"
SEL_MORE_BTN     = 'button.w8nwRe[aria-expanded="false"]'
SEL_REVIEWER_BTN = "button.al6Kxe"
SEL_REVIEWER_NAME= "div.d4r55"
SEL_REVIEWER_SUB = "div.RfnDt"
SEL_RATING       = 'span.kvMYJc[role="img"]'
SEL_TIMESTAMP    = "span.rsqaWe"
SEL_REVIEW_TEXT  = "span.wiI7pd"
SEL_META_BLOCK   = "div.PBK6be"
SEL_META_SPAN    = "span.RfDO5c"
SEL_PHOTO_BTN    = "button.Tya61d"

SCROLL_PANEL_SELS = [
    "div.m6QErb.DxyBCb",
    "div.m6QErb[aria-label]",
    'div[role="feed"]',
]


# ── One-time session setup ────────────────────────────────────────────────────

async def setup_session():
    """
    One-time setup: copy your real Chrome profile to a temp dir (avoids the
    SingletonLock so Chrome can be open), launch it, visit Google Maps, and
    save Google cookies to data/.maps_cookies.json.

    Chrome does NOT need to be closed — we work from a copy.
    """
    import shutil
    import tempfile

    log.info("=== One-time session setup ===")
    log.info(f"Source Chrome profile: {REAL_CHROME_PROFILE}")

    if not REAL_CHROME_PROFILE.exists():
        raise FileNotFoundError(f"Chrome profile not found: {REAL_CHROME_PROFILE}")

    with tempfile.TemporaryDirectory(prefix="chrome_setup_") as tmp_dir:
        tmp_profile = Path(tmp_dir)
        log.info(f"Copying profile to temp dir: {tmp_profile}")

        # Copy Default directory — skip large caches and lock files
        src_default = REAL_CHROME_PROFILE / "Default"
        dst_default = tmp_profile / "Default"
        if src_default.exists():
            shutil.copytree(
                src_default,
                dst_default,
                ignore=shutil.ignore_patterns(
                    "Cache", "Code Cache", "GPUCache", "DawnGraphiteCache",
                    "DawnWebGPUCache", "ShaderCache", "Crashpad", "*.log",
                    "Singleton*",
                ),
            )
            log.info("Profile Default directory copied")

        # Copy Local State (holds encryption key metadata)
        local_state = REAL_CHROME_PROFILE / "Local State"
        if local_state.exists():
            shutil.copy2(local_state, tmp_profile / "Local State")

        async with async_playwright() as p:
            log.info("Launching Chrome with copied profile ...")
            ctx = await p.chromium.launch_persistent_context(
                user_data_dir=str(tmp_profile),
                channel="chrome",
                headless=False,
                args=["--no-first-run", "--disable-default-apps", "--disable-session-crashed-bubble"],
                viewport={"width": 1280, "height": 800},
            )
            page = await ctx.new_page()
            log.info("Navigating to Google Maps ...")
            await page.goto("https://www.google.com/maps", wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(5000)
            log.info(f"Loaded: {page.url}")

            all_cookies = await ctx.cookies()
            google_cookies = [c for c in all_cookies if "google" in c.get("domain", "")]
            log.info(f"Extracted {len(google_cookies)} Google cookies")

            COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(COOKIES_FILE, "w") as f:
                json.dump(google_cookies, f, indent=2)

            log.info(f"Cookies saved to {COOKIES_FILE}")
            await ctx.close()


# ── Parsing helpers ───────────────────────────────────────────────────────────

async def _text(loc: Locator, default: str = "") -> str:
    try:
        if await loc.count() > 0:
            return ((await loc.first.text_content()) or default).strip()
    except Exception:
        pass
    return default


async def _attr(loc: Locator, attr: str, default=None):
    try:
        if await loc.count() > 0:
            return (await loc.first.get_attribute(attr)) or default
    except Exception:
        pass
    return default


async def parse_reviewer(review_loc: Locator) -> dict:
    btn = review_loc.locator(SEL_REVIEWER_BTN)
    if await btn.count() == 0:
        return {}

    name     = await _text(btn.locator(SEL_REVIEWER_NAME))
    sub_text = await _text(btn.locator(SEL_REVIEWER_SUB))
    href     = (await _attr(btn, "data-href")) or ""

    m  = re.search(r"/contrib/(\d+)/", href)
    rc = re.search(r"([\d,]+)\s+reviews?", sub_text)
    pc = re.search(r"([\d,]+)\s+photos?",  sub_text)

    return {
        "reviewer_name":    name,
        "contributor_id":   m.group(1) if m else None,
        "profile_url":      href,
        "is_local_guide":   "Local Guide" in sub_text,
        "reviewer_reviews": int(rc.group(1).replace(",", "")) if rc else None,
        "reviewer_photos":  int(pc.group(1).replace(",", "")) if pc else None,
    }


async def parse_rating(review_loc: Locator):
    span = review_loc.locator(SEL_RATING)
    if await span.count() == 0:
        return None
    aria = (await span.first.get_attribute("aria-label")) or ""
    m = re.search(r"(\d+)", aria)
    return int(m.group(1)) if m else None


async def parse_meta(review_loc: Locator) -> dict:
    meta = {}
    for block in await review_loc.locator(SEL_META_BLOCK).all():
        texts = []
        for s in await block.locator(SEL_META_SPAN).all():
            t = await s.text_content()
            if t and t.strip():
                texts.append(t.strip())
        if not texts:
            continue
        if len(texts) == 1 and ":" in texts[0]:
            k, _, v = texts[0].partition(":")
            meta[k.strip()] = v.strip()
        elif len(texts) >= 2:
            k, v = texts[0], texts[1]
            if k in meta:
                meta[k] = meta[k] if isinstance(meta[k], list) else [meta[k]]
                meta[k].append(v)
            else:
                meta[k] = v
    return meta


async def parse_review(review_loc: Locator) -> dict:
    return {
        "review_id":       await review_loc.get_attribute("data-review-id"),
        **await parse_reviewer(review_loc),
        "rating":          await parse_rating(review_loc),
        "timestamp":       await _text(review_loc.locator(SEL_TIMESTAMP)),
        "text":            await _text(review_loc.locator(SEL_REVIEW_TEXT)),
        "meta":            await parse_meta(review_loc),
        "attached_photos": await review_loc.locator(SEL_PHOTO_BTN).count(),
    }


# ── Scrolling ─────────────────────────────────────────────────────────────────

async def _find_scroll_panel_sel(page: Page) -> str | None:
    for sel in SCROLL_PANEL_SELS:
        if await page.locator(sel).count() > 0:
            log.debug(f"Scroll panel found: {sel}")
            return sel
        log.debug(f"Scroll panel not found: {sel}")
    return None


async def _scroll_until(page: Page, panel_sel: str, target: int, timeout: int = 120):
    log.info(f"Scrolling to load {target} reviews ...")
    deadline = time.time() + timeout
    stall = prev = 0
    while time.time() < deadline:
        count = await page.locator(SEL_REVIEW_ITEM).count()
        log.debug(f"  {count} reviews loaded")
        if count >= target:
            log.info(f"Target reached: {count}")
            break
        if count == prev:
            stall += 1
            if stall >= 6:
                log.info(f"No more reviews loading — stopped at {count}")
                break
            # Wait longer when stalled to give Maps time to fetch the next batch
            wait_ms = 3000 if stall >= 2 else 2000
        else:
            stall = 0
            wait_ms = 2000
        prev = count
        await page.evaluate(
            "(sel) => { const el = document.querySelector(sel); if (el) el.scrollTop = el.scrollHeight; }",
            panel_sel,
        )
        await page.wait_for_timeout(wait_ms)


# ── Main scraping function ────────────────────────────────────────────────────

async def scrape_reviews(place_id: str, target: int = 100, headless: bool = False) -> list:
    url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    reviews = []

    log.info(f"Scraping place_id={place_id}")
    log.info(f"URL: {url}")

    if not COOKIES_FILE.exists():
        log.warning(
            f"No saved cookies found at {COOKIES_FILE}. "
            "Run await setup_session() first for best results."
        )

    async with AsyncCamoufox(headless=headless, os="linux") as browser:
        log.info("Camoufox browser launched ...")
        page = await browser.new_page()

        # Inject saved Google cookies so Maps sees a returning, trusted user
        if COOKIES_FILE.exists():
            with open(COOKIES_FILE) as f:
                cookies = json.load(f)
            await page.context.add_cookies(cookies)
            log.info(f"Injected {len(cookies)} saved Google cookies")

        log.info("Navigating to Maps ...")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as e:
            log.warning(f"goto raised (continuing anyway): {e}")

        # Wait for Maps to redirect to the full place URL
        try:
            await page.wait_for_url("**/maps/place/**", timeout=10_000)
            log.info("Redirected to full place URL")
        except Exception:
            log.warning("No redirect detected — continuing with current URL")

        # Wait for Reviews tab to appear in the DOM
        log.info("Waiting for Reviews tab ...")
        try:
            await page.wait_for_selector(f'{SEL_REVIEWS_TAB}:has-text("Reviews")', timeout=15_000)
            log.info("Reviews tab visible")
        except Exception:
            log.warning("Reviews tab not found after 15s — still trying ...")
        await page.wait_for_timeout(2000)

        log.info(f"Current URL : {page.url}")
        log.info(f"Page title  : {await page.title()}")

        # Find and click Reviews tab
        log.info("Looking for Reviews tab ...")
        tabs = await page.locator(SEL_REVIEWS_TAB).all()
        log.debug(f"Found {len(tabs)} tab elements")
        for i, tab in enumerate(tabs):
            log.debug(f"  Tab {i}: '{await tab.text_content()}'")

        clicked = False
        for tab in tabs:
            if (await tab.text_content() or "").strip() == "Reviews":
                log.info("Clicking Reviews tab ...")
                await tab.click()
                clicked = True
                break

        if not clicked:
            log.warning("Reviews tab not found — page may be a stripped/bot version")
            log.debug(f"Page source snippet:\n{(await page.content())[:600]}")
            await browser.close()
            raise NoReviewsTabError()

        log.info("Reviews tab clicked — waiting 2s ...")
        await page.wait_for_timeout(2000)

        # Find scrollable panel
        log.info("Looking for scrollable reviews panel ...")
        panel_sel = await _find_scroll_panel_sel(page)
        if panel_sel is None:
            log.warning("Scrollable panel not found — possible bot detection")
            await browser.close()
            raise BotDetectionError()

        # Scroll to load reviews
        await _scroll_until(page, panel_sel, target)

        # Expand truncated reviews
        more_btns = page.locator(SEL_MORE_BTN)
        n_more = await more_btns.count()
        log.info(f"Expanding {n_more} truncated reviews ...")
        await more_btns.evaluate_all("btns => btns.forEach(b => b.click())")
        if n_more:
            await page.wait_for_timeout(500)

        # Parse reviews
        review_locs = await page.locator(SEL_REVIEW_ITEM).all()
        log.info(f"Parsing {min(len(review_locs), target)} of {len(review_locs)} reviews ...")
        for loc in review_locs[:target]:
            try:
                reviews.append(await parse_review(loc))
            except Exception as e:
                log.error(f"Parse error: {e}")

        await browser.close()

    log.info(f"Done — {len(reviews)} reviews for {place_id}")
    return reviews


# ── Batch runner ──────────────────────────────────────────────────────────────

async def run_batch(
    restaurants_dir: Path = RESTAURANTS_DIR,
    reviews_dir:     Path = REVIEWS_DIR,
    target:  int  = 100,
    limit:   int  = None,
    headless: bool = False,
):
    json_files = sorted(restaurants_dir.glob("*.json"))
    if limit:
        json_files = json_files[:limit]

    for fpath in json_files:
        with open(fpath) as f:
            data = json.load(f)

        restaurants = data.get("restaurants", [])
        cbg = data.get("cbg", fpath.stem)
        log.info(f"=== CBG {cbg} — {len(restaurants)} restaurants ===")

        for rest in restaurants:
            place_id = rest.get("id")
            name     = rest.get("displayName", {}).get("text", place_id)
            out_file = reviews_dir / f"{place_id}.json"

            if out_file.exists():
                log.info(f"SKIP {name}")
                continue

            log.info(f"Scraping {name} ({place_id}) ...")
            try:
                reviews = await scrape_reviews(place_id, target=target, headless=headless)
            except Exception as e:
                log.error(f"Failed: {e}")
                reviews = []

            payload = {
                "place_id":      place_id,
                "name":          name,
                "cbg":           cbg,
                "scraped_at":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total_reviews": len(reviews),
                "reviews":       reviews,
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            await asyncio.sleep(2)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Google Maps reviews")
    parser.add_argument("--setup",    action="store_true", help="One-time cookie extraction from real Chrome profile")
    parser.add_argument("--place-id", help="Scrape a single place by ID")
    parser.add_argument("--target",   type=int, default=100)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(setup_session())
    elif args.place_id:
        result = asyncio.run(scrape_reviews(args.place_id, target=args.target, headless=args.headless))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        asyncio.run(run_batch(target=args.target, limit=args.limit, headless=args.headless))
