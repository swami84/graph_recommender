#!/usr/bin/env python3
"""Project-private Camoufox cache plus a cross-process launch lock.

Import this module *before* anything imports ``camoufox``.

Background: ``camoufox`` resolves its browser install to
``platformdirs.user_cache_dir("camoufox")`` -- by default the shared
``~/.cache/camoufox``.  Any other project on this machine running a
different camoufox major version will ``shutil.rmtree`` that whole
directory on launch to migrate the layout, yanking the browser binaries
out from under our running shards mid-scrape.  Pointing XDG_CACHE_HOME at
a project-private root gives this scraper an install nobody else touches.
"""

from __future__ import annotations

import fcntl
import os
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Project-private XDG cache root; camoufox installs to <root>/camoufox.
CACHE_ROOT = Path(
    os.environ.get("FOODIE_CAMOUFOX_CACHE_ROOT", ROOT / ".cache")
).resolve()

# Must be set before camoufox is imported: its submodules compute
# INSTALL_DIR at module import time.
if "camoufox" in sys.modules:  # pragma: no cover - defensive
    raise RuntimeError(
        "camoufox_env must be imported before camoufox; the shared cache "
        "would already be resolved."
    )
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

INSTALL_DIR = CACHE_ROOT / "camoufox"

# Serializes browser startup across every shard process.  Camoufox startup
# reads (and, on a version change, rewrites) the shared install directory;
# holding this lock means only one process is ever inside that window.
LAUNCH_LOCK_FILE = CACHE_ROOT / "camoufox_launch.lock"

LAUNCH_LOCK_TIMEOUT = float(os.environ.get("FOODIE_CAMOUFOX_LOCK_TIMEOUT", "300"))


@contextmanager
def _flock(path: Path, timeout: float):
    """Hold an exclusive advisory lock on ``path``, or raise on timeout."""
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with open(path, "a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout:g}s waiting for the "
                        f"Camoufox launch lock at {path}"
                    )
                time.sleep(0.25)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def assert_installed() -> Path:
    """Fail fast with a repair hint if the private install is missing."""
    if not (INSTALL_DIR / "version.json").exists():
        raise FileNotFoundError(
            f"No Camoufox install at {INSTALL_DIR}. Repair it with:\n"
            f"  XDG_CACHE_HOME={CACHE_ROOT} python -m camoufox fetch"
        )
    return INSTALL_DIR


@asynccontextmanager
async def open_camoufox(**kwargs):
    """``AsyncCamoufox`` with startup serialized across shard processes.

    The lock covers only browser startup, not the scraping session, so the
    shards still run fully concurrent once launched.
    """
    import asyncio

    from camoufox.async_api import AsyncCamoufox

    session = AsyncCamoufox(**kwargs)
    # flock is blocking; keep the event loop responsive while we queue.
    with_lock = _flock(LAUNCH_LOCK_FILE, LAUNCH_LOCK_TIMEOUT)
    await asyncio.to_thread(with_lock.__enter__)
    try:
        browser = await session.__aenter__()
    finally:
        await asyncio.to_thread(with_lock.__exit__, None, None, None)

    try:
        yield browser
    except BaseException:
        await session.__aexit__(*sys.exc_info())
        raise
    else:
        await session.__aexit__(None, None, None)
