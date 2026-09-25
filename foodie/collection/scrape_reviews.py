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
    python -m foodie.collection.scrape_reviews --setup          # one-time cookie extraction
    python -m foodie.collection.scrape_reviews --place-id ChIJM3elBidawokRJKPC55N6pQI
    python -m foodie.collection.scrape_reviews --limit 3 --headless
"""

import argparse
import asyncio
import datetime
import json
import logging
import random
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from foodie.collection import camoufox_env  # must precede any camoufox import
from foodie.collection.camoufox_env import open_camoufox
from playwright.async_api import (
    async_playwright,
    BrowserContext,
    Page,
    Locator,
)

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


class GoogleChallengeError(BotDetectionError):
    """Google has presented its IP-wide automated-traffic challenge page."""

# ── Paths ─────────────────────────────────────────────────────────────────────
RESTAURANTS_DIR   = Path("data/restaurants")
REVIEWS_DIR       = Path("data/reviews")
COOKIES_FILE      = Path("data/.maps_cookies.json")   # saved Google cookies
REAL_CHROME_PROFILE = Path.home() / ".config/google-chrome"
CAPTCHA_TEST_PLACE_ID = "ChIJye4iijK3t4kR1e9LNbNnqms"

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


async def solve_captcha_session(place_id: str) -> bool:
    """Open an interactive Camoufox session and persist cleared cookies."""
    url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    log.info("Opening headed Camoufox CAPTCHA session ...")
    async with open_camoufox(headless=False, os="linux") as browser:
        context = await browser.new_context()
        try:
            await initialize_maps_context(context)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            log.info(f"Opened: {page.url}")
            log.info(
                "Solve the CAPTCHA and wait until the normal Maps place page "
                "and Reviews tab are visible."
            )

            while True:
                answer = await asyncio.to_thread(
                    input,
                    "Press Enter after Maps is restored, or type q to abort: ",
                )
                if answer.strip().lower() == "q":
                    log.warning("CAPTCHA session closed without saving cookies")
                    return False
                if "google.com/sorry/" in page.url.lower():
                    log.warning(
                        "The page is still on Google's challenge. Complete it "
                        "before pressing Enter."
                    )
                    continue
                break

            google_cookies = [
                cookie for cookie in await context.cookies()
                if "google" in cookie.get("domain", "")
            ]
            COOKIES_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary = COOKIES_FILE.with_name(f".{COOKIES_FILE.name}.tmp")
            temporary.write_text(
                json.dumps(google_cookies, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(COOKIES_FILE)
            log.info(
                f"Saved {len(google_cookies)} updated Google cookies to "
                f"{COOKIES_FILE}"
            )
            return True
        finally:
            await context.close()


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


async def parse_reviews_bulk(page: Page, target: int) -> list[dict]:
    """Extract all visible review cards in one browser round trip."""
    selectors = {
        "item": SEL_REVIEW_ITEM,
        "reviewer_btn": SEL_REVIEWER_BTN,
        "reviewer_name": SEL_REVIEWER_NAME,
        "reviewer_sub": SEL_REVIEWER_SUB,
        "rating": SEL_RATING,
        "timestamp": SEL_TIMESTAMP,
        "text": SEL_REVIEW_TEXT,
        "meta_block": SEL_META_BLOCK,
        "meta_span": SEL_META_SPAN,
        "photo_btn": SEL_PHOTO_BTN,
    }
    return await page.locator(SEL_REVIEW_ITEM).evaluate_all(
        r"""(nodes, options) => {
            const {selectors: s, target} = options;
            const text = (root, selector) =>
                (root?.querySelector(selector)?.textContent || '').trim();
            const numberFrom = (value, pattern) => {
                const match = (value || '').match(pattern);
                return match ? Number(match[1].replaceAll(',', '')) : null;
            };

            return nodes.slice(0, target).map(node => {
                const reviewerButton = node.querySelector(s.reviewer_btn);
                const reviewerSub = text(reviewerButton, s.reviewer_sub);
                const profileUrl = reviewerButton?.getAttribute('data-href') || '';
                const contributorMatch = profileUrl.match(/\/contrib\/(\d+)\//);
                const ratingLabel =
                    node.querySelector(s.rating)?.getAttribute('aria-label') || '';
                const meta = {};

                for (const block of node.querySelectorAll(s.meta_block)) {
                    const values = [...block.querySelectorAll(s.meta_span)]
                        .map(span => (span.textContent || '').trim())
                        .filter(Boolean);
                    if (!values.length) continue;
                    if (values.length === 1 && values[0].includes(':')) {
                        const separator = values[0].indexOf(':');
                        meta[values[0].slice(0, separator).trim()] =
                            values[0].slice(separator + 1).trim();
                    } else if (values.length >= 2) {
                        const [key, value] = values;
                        if (Object.hasOwn(meta, key)) {
                            meta[key] = Array.isArray(meta[key])
                                ? [...meta[key], value]
                                : [meta[key], value];
                        } else {
                            meta[key] = value;
                        }
                    }
                }

                return {
                    review_id: node.getAttribute('data-review-id'),
                    reviewer_name: text(reviewerButton, s.reviewer_name),
                    contributor_id: contributorMatch ? contributorMatch[1] : null,
                    profile_url: profileUrl,
                    is_local_guide: reviewerSub.includes('Local Guide'),
                    reviewer_reviews: numberFrom(
                        reviewerSub, /([\d,]+)\s+reviews?/i
                    ),
                    reviewer_photos: numberFrom(
                        reviewerSub, /([\d,]+)\s+photos?/i
                    ),
                    rating: numberFrom(ratingLabel, /(\d+)/),
                    timestamp: text(node, s.timestamp),
                    text: text(node, s.text),
                    meta,
                    attached_photos: node.querySelectorAll(s.photo_btn).length,
                };
            });
        }""",
        {"selectors": selectors, "target": target},
    )


# ── Scrolling ─────────────────────────────────────────────────────────────────

async def _find_scroll_panel_sel(page: Page) -> str | None:
    for sel in SCROLL_PANEL_SELS:
        if await page.locator(sel).count() > 0:
            log.debug(f"Scroll panel found: {sel}")
            return sel
        log.debug(f"Scroll panel not found: {sel}")
    return None


async def _displayed_review_count(page: Page) -> int | None:
    """Read the venue's review total from the rating-summary controls."""
    selectors = [
        'button[jsaction*="reviewChart"]',
        'div.F7nice button',
        'div.F7nice span[aria-label*="review" i]',
        'div.F7nice',
    ]
    for selector in selectors:
        locator = page.locator(selector)
        for index in range(await locator.count()):
            item = locator.nth(index)
            try:
                values = [
                    await item.get_attribute("aria-label"),
                    await item.get_attribute("title"),
                    await item.text_content(),
                ]
            except Exception:
                continue
            for value in values:
                match = re.search(r"([\d,]+)\s+(?:Google\s+)?reviews?\b", value or "", re.I)
                if match:
                    return int(match.group(1).replace(",", ""))
    return None


async def _activate_reviews_panel(page: Page) -> str | None:
    """Scroll the Reviews view until Maps materializes virtualized cards."""
    panel_sel = None
    for attempt in range(6):
        # Some Maps layouts do not populate the feed until the newly opened
        # Reviews tab receives an actual downward wheel event.
        await page.mouse.wheel(0, 900)
        panel_sel = await _find_scroll_panel_sel(page)
        if panel_sel is None:
            try:
                await page.wait_for_selector(
                    ", ".join(SCROLL_PANEL_SELS), timeout=1_000
                )
            except Exception:
                continue
            panel_sel = await _find_scroll_panel_sel(page)
        if panel_sel:
            panel = page.locator(panel_sel).first
            try:
                await panel.hover()
                await page.mouse.wheel(0, 900)
            except Exception:
                pass
            await page.evaluate(
                """(sel) => {
                    const el = document.querySelector(sel);
                    if (el) {
                        el.scrollTop = Math.max(el.scrollTop + 900, 900);
                        el.dispatchEvent(new Event('scroll', {bubbles: true}));
                    }
                }""",
                panel_sel,
            )
            try:
                await page.wait_for_selector(SEL_REVIEW_ITEM, timeout=1_500)
            except Exception:
                continue
            if await page.locator(SEL_REVIEW_ITEM).count() > 0:
                log.info(
                    f"Reviews feed activated after {attempt + 1} scroll attempt(s)"
                )
                return panel_sel
    return panel_sel


async def _scroll_until(page: Page, panel_sel: str, target: int, timeout: int = 90):
    log.info(f"Scrolling to load {target} reviews ...")
    deadline = time.time() + timeout
    stall = 0
    while time.time() < deadline:
        count = await page.locator(SEL_REVIEW_ITEM).count()
        log.debug(f"  {count} reviews loaded")
        if count >= target:
            log.info(f"Target reached: {count}")
            break
        await page.evaluate(
            """(sel) => {
                const el = document.querySelector(sel);
                if (!el) return;
                el.scrollTop = el.scrollHeight;
                el.dispatchEvent(new Event('scroll', {bubbles: true}));
            }""",
            panel_sel,
        )
        try:
            await page.wait_for_function(
                "([sel, previous]) => document.querySelectorAll(sel).length > previous",
                [SEL_REVIEW_ITEM, count],
                timeout=2_000 if stall == 0 else 3_000,
            )
            stall = 0
            # Avoid issuing successive review-pagination requests in a tight,
            # machine-like loop even when Maps responds immediately.
            await page.wait_for_timeout(random.randint(800, 1_500))
        except Exception:
            stall += 1
            if stall >= 3:
                log.info(f"No more reviews loading — stopped at {count}")
                break


async def initialize_maps_context(context: BrowserContext) -> None:
    """Seed one long-lived worker context with the saved Google session."""
    if COOKIES_FILE.exists():
        with open(COOKIES_FILE) as cookie_file:
            cookies = json.load(cookie_file)
        await context.add_cookies(cookies)
        log.info(f"Injected {len(cookies)} saved Google cookies into worker context")


@asynccontextmanager
async def _fresh_page(
    headless: bool,
    context: BrowserContext | None = None,
):
    """Open a fresh page while preserving a worker's browser session."""
    if context is None:
        async with open_camoufox(headless=headless, os="linux") as owned_browser:
            owned_context = await owned_browser.new_context()
            try:
                await initialize_maps_context(owned_context)
                async with _fresh_page(headless, owned_context) as page:
                    yield page
            finally:
                await owned_context.close()
        return

    page = await context.new_page()
    try:
        yield page
    finally:
        await page.close()


# ── Main scraping function ────────────────────────────────────────────────────

async def scrape_reviews(
    place_id: str,
    target: int = 100,
    headless: bool = False,
    context: BrowserContext | None = None,
) -> list:
    url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
    reviews = []

    log.info(f"Scraping place_id={place_id}")
    log.info(f"URL: {url}")

    if not COOKIES_FILE.exists():
        log.warning(
            f"No saved cookies found at {COOKIES_FILE}. "
            "Run await setup_session() first for best results."
        )

    async with _fresh_page(headless, context) as page:
        log.info("Fresh Maps page created in stable worker context ...")

        log.info("Navigating to Maps ...")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as e:
            # One navigation attempt per run. Preserve any failure as retryable
            # instead of allowing an offline page to become no_reviews_tab.
            raise RuntimeError(
                f"Google Maps navigation failed; retry required: {e}"
            ) from e

        # Wait for Maps to redirect to the full place URL
        try:
            await page.wait_for_url("**/maps/place/**", timeout=10_000)
            log.info("Redirected to full place URL")
        except Exception:
            log.warning("No redirect detected — continuing with current URL")

        # Wait for Reviews tab to appear in the DOM
        log.info("Waiting for Reviews tab ...")
        reviews_tab_visible = False
        try:
            await page.wait_for_selector(f'{SEL_REVIEWS_TAB}:has-text("Reviews")', timeout=15_000)
            log.info("Reviews tab visible")
            reviews_tab_visible = True
        except Exception:
            log.warning("Reviews tab not found after 15s — still trying ...")
        current_url = page.url
        page_title = await page.title()
        log.info(f"Current URL : {current_url}")
        log.info(f"Page title  : {page_title}")

        if "google.com/sorry/" in current_url.lower():
            raise GoogleChallengeError(
                "Google automated-traffic challenge page detected"
            )

        network_failure_markers = (
            "about:neterror",
            "server not found",
            "problem loading page",
            "network error",
            "you're offline",
            "you are offline",
        )
        page_identity = f"{current_url} {page_title}".lower()
        if not reviews_tab_visible and any(
            marker in page_identity for marker in network_failure_markers
        ):
            raise RuntimeError(
                "Google Maps navigation failed before the Reviews tab loaded; "
                f"retry required: {page_identity}"
            )

        # Find and click Reviews tab
        log.info("Looking for Reviews tab ...")
        displayed_count = await _displayed_review_count(page)
        effective_target = min(target, displayed_count) if displayed_count else target
        if displayed_count is not None:
            log.info(
                f"Displayed review count={displayed_count}; "
                f"collection target={effective_target}"
            )
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
            # A genuine zero-review business still renders a normal Maps place
            # page with its business heading.  Without that positive signal,
            # an absent Reviews tab is ambiguous (offline, partial, or stripped)
            # and must remain retryable instead of becoming a false no-review.
            place_heading = page.locator("h1.DUwDvf")
            heading_text = await _text(place_heading)
            if not heading_text:
                raise RuntimeError(
                    "Reviews tab absent and Google Maps place page could not be "
                    "validated; retry required"
                )
            log.warning(
                f"Reviews tab not found on validated place page '{heading_text}'"
            )
            log.debug(f"Page source snippet:\n{(await page.content())[:600]}")
            raise NoReviewsTabError()

        log.info("Reviews tab clicked — activating review feed ...")

        # Maps may defer creation/loading of the virtualized review feed until
        # the Reviews view is scrolled. Activate it before requiring cards.
        log.info("Scrolling Reviews view to activate the reviews panel ...")
        panel_sel = await _activate_reviews_panel(page)
        if panel_sel is None:
            log.warning("Scrollable panel not found — possible bot detection")
            raise BotDetectionError()

        # A visible Reviews tab is positive evidence that at least one review
        # should load.  Previously, a slow/partial panel could reach the parser
        # with zero review cards and be committed as a successful zero-review
        # scrape. Keep that ambiguous state retryable instead.
        try:
            await page.wait_for_selector(SEL_REVIEW_ITEM, timeout=10_000)
        except Exception as e:
            raise RuntimeError(
                "Reviews tab opened but no review cards loaded; retry required"
            ) from e

        # Scroll to load reviews
        await _scroll_until(page, panel_sel, effective_target)

        # Expand truncated reviews
        more_btns = page.locator(SEL_MORE_BTN)
        n_more = await more_btns.count()
        log.info(f"Expanding {n_more} truncated reviews ...")
        await more_btns.evaluate_all("btns => btns.forEach(b => b.click())")
        if n_more:
            try:
                await page.wait_for_function(
                    "sel => document.querySelectorAll(sel).length === 0",
                    SEL_MORE_BTN,
                    timeout=1_500,
                )
            except Exception:
                # Expansion is best-effort; text already present in the DOM is
                # still useful if Maps does not update the button state.
                pass

        # Parse reviews
        review_count = await page.locator(SEL_REVIEW_ITEM).count()
        if not review_count:
            raise RuntimeError(
                "Reviews tab opened but zero review cards were available at "
                "parse time; retry required"
            )
        parse_count = min(review_count, effective_target)
        log.info(f"Bulk parsing {parse_count} of {review_count} reviews ...")
        try:
            reviews = await parse_reviews_bulk(page, parse_count)
        except Exception as e:
            log.warning(f"Bulk parse failed; using compatibility parser: {e}")
            review_locs = await page.locator(SEL_REVIEW_ITEM).all()
            for loc in review_locs[:parse_count]:
                try:
                    reviews.append(await parse_review(loc))
                except Exception as parse_error:
                    log.error(f"Parse error: {parse_error}")

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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--setup", action="store_true", help="One-time cookie extraction from real Chrome profile")
    mode.add_argument(
        "--solve-captcha",
        action="store_true",
        help="Open headed Camoufox, wait for manual CAPTCHA solving, and save cookies",
    )
    parser.add_argument("--place-id", help="Scrape a single place by ID")
    parser.add_argument("--target",   type=int, default=100)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.setup:
        asyncio.run(setup_session())
    elif args.solve_captcha:
        solved = asyncio.run(solve_captcha_session(
            args.place_id or CAPTCHA_TEST_PLACE_ID
        ))
        if not solved:
            raise SystemExit(1)
    elif args.place_id:
        try:
            result = asyncio.run(scrape_reviews(
                args.place_id, target=args.target, headless=args.headless
            ))
        except GoogleChallengeError as exc:
            log.error(str(exc))
            raise SystemExit(76)
        except Exception as exc:
            log.error(f"Known-positive probe failed: {exc}")
            raise SystemExit(1)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        asyncio.run(run_batch(target=args.target, limit=args.limit, headless=args.headless))
