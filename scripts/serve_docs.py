#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Build Volta's rustdoc and serve it over HTTP, for reading the docs from another machine.

Usage (from anywhere; paths resolve against the repository this script lives in):
    uv run scripts/serve_docs.py                          # build, serve on 127.0.0.1:8000
    uv run scripts/serve_docs.py --host 0.0.0.0           # reachable from the LAN
    uv run scripts/serve_docs.py --features gpu,accelerate
    uv run scripts/serve_docs.py --rebuild-every 600 --pull
    uv run scripts/serve_docs.py --no-build               # serve the last snapshot as is
    uv run scripts/serve_docs.py -- --document-private-items   # extra `cargo doc` args

What it does, in order:
1. `cargo doc --no-deps` (default features, which are CPU-only) into the usual `target/`.
2. Copies `target/doc` to a fresh snapshot directory under `target/doc-serve/`.
3. Serves that snapshot with the standard library's threaded HTTP server. `/` redirects
   to `/volta/`; directory listings are off.

With `--rebuild-every N`, it rebuilds every N seconds (after `git pull --ff-only` when
`--pull` is given) and swaps to the new snapshot only when the build succeeds, so a
reader never sees a half-written tree or a failed build. A failed pull or build is logged
and the previous snapshot keeps being served.

No authentication and no TLS: by default it binds to 127.0.0.1. To reach it from outside
the machine, bind to a private interface (LAN, Tailscale) or put a reverse proxy that
handles TLS and auth (Caddy, nginx) in front and keep this on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import http.server
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("serve_docs")

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE = "volta"
KEEP_SNAPSHOTS = 2  # the live one plus the previous one, for requests still in flight


class Snapshot:
    """The directory currently being served; swapped atomically under a lock."""

    def __init__(self, directory: Path | None) -> None:
        self._lock = threading.Lock()
        self._directory = directory

    def get(self) -> Path | None:
        with self._lock:
            return self._directory

    def set(self, directory: Path) -> None:
        with self._lock:
            self._directory = directory


def cargo_doc(target_dir: Path, features: str | None, extra: list[str]) -> bool:
    cmd = ["cargo", "doc", "--no-deps", "--target-dir", str(target_dir)]
    if features:
        cmd += ["--features", features]
    cmd += extra
    log.info("building: %s", " ".join(cmd))
    started = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        log.error("cargo doc failed (exit %d)", result.returncode)
        return False
    log.info("built in %.1fs", time.monotonic() - started)
    return True


def git_pull() -> bool:
    result = subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        log.error("git pull --ff-only failed (exit %d)", result.returncode)
        return False
    return True


def take_snapshot(doc_dir: Path, snapshots_dir: Path) -> Path:
    """Copy the built docs into a new snapshot directory and prune old ones."""
    if not (doc_dir / CRATE / "index.html").is_file():
        raise FileNotFoundError(
            f"{doc_dir / CRATE / 'index.html'} is missing after the build"
        )
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshots_dir / time.strftime("%Y%m%dT%H%M%S")
    suffix = 0
    while dest.exists():  # two builds in one second
        suffix += 1
        dest = snapshots_dir / f"{time.strftime('%Y%m%dT%H%M%S')}-{suffix}"
    tmp = dest.with_name(dest.name + ".partial")
    shutil.copytree(doc_dir, tmp)
    tmp.rename(dest)
    for old in sorted(p for p in snapshots_dir.iterdir() if p.is_dir())[
        :-KEEP_SNAPSHOTS
    ]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def latest_snapshot(snapshots_dir: Path) -> Path | None:
    if not snapshots_dir.is_dir():
        return None
    done = sorted(
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir() and not p.name.endswith(".partial")
    )
    return done[-1] if done else None


def make_handler(snapshot: Snapshot) -> type[http.server.SimpleHTTPRequestHandler]:
    class DocsHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            directory = snapshot.get()
            # A snapshot always exists before the server starts; see main().
            assert directory is not None
            super().__init__(*args, directory=str(directory), **kwargs)

        def _redirect_root(self) -> bool:
            if self.path.split("?", 1)[0] not in ("/", "/index.html"):
                return False
            self.send_response(302)
            self.send_header("Location", f"/{CRATE}/")
            self.end_headers()
            return True

        def do_GET(self) -> None:
            if not self._redirect_root():
                super().do_GET()

        def do_HEAD(self) -> None:
            if not self._redirect_root():
                super().do_HEAD()

        def list_directory(self, path: str | os.PathLike[str]) -> None:
            self.send_error(404, "Not found")

        def end_headers(self) -> None:
            # Snapshots change under the same URLs; make browsers revalidate.
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, format: str, *args: object) -> None:
            log.info("%s %s", self.address_string(), format % args)

    return DocsHandler


def rebuild_loop(
    every: float,
    pull: bool,
    build: Callable[[], bool],
    snapshot: Snapshot,
    doc_dir: Path,
    snapshots_dir: Path,
    stop: threading.Event,
) -> None:
    while not stop.wait(every):
        if pull and not git_pull():
            continue
        if not build():
            continue
        try:
            snapshot.set(take_snapshot(doc_dir, snapshots_dir))
            log.info("now serving %s", snapshot.get())
        except OSError as e:
            log.error("snapshot failed, still serving the previous one: %s", e)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Volta's rustdoc and serve it over HTTP.",
        epilog="Arguments after `--` go to `cargo doc` verbatim.",
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)"
    )
    p.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    p.add_argument(
        "--features",
        help="cargo features to document, e.g. gpu,accelerate (default: none, CPU-only)",
    )
    p.add_argument(
        "--target-dir",
        type=Path,
        default=REPO_ROOT / "target",
        help="cargo target directory (default: <repo>/target)",
    )
    p.add_argument(
        "--no-build",
        action="store_true",
        help="skip the initial build and serve the latest snapshot",
    )
    p.add_argument(
        "--rebuild-every",
        type=float,
        metavar="SECONDS",
        help="rebuild periodically and swap in the new docs on success",
    )
    p.add_argument(
        "--pull",
        action="store_true",
        help="run `git pull --ff-only` before each periodic rebuild (needs --rebuild-every)",
    )
    p.add_argument("cargo_args", nargs="*", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.pull and args.rebuild_every is None:
        p.error("--pull needs --rebuild-every")
    if args.rebuild_every is not None and args.rebuild_every <= 0:
        p.error("--rebuild-every must be positive")
    if not 0 < args.port < 65536:
        p.error("--port must be in 1..65535")
    return args


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    target_dir: Path = args.target_dir.resolve()
    doc_dir = target_dir / "doc"
    snapshots_dir = target_dir / "doc-serve"

    def build() -> bool:
        return cargo_doc(target_dir, args.features, args.cargo_args)

    if args.no_build:
        current = latest_snapshot(snapshots_dir)
        if current is None:
            log.error(
                "--no-build: no snapshot in %s; run once without it", snapshots_dir
            )
            return 1
    else:
        if not build():
            return 1
        current = take_snapshot(doc_dir, snapshots_dir)
    snapshot = Snapshot(current)

    server = http.server.ThreadingHTTPServer(
        (args.host, args.port), make_handler(snapshot)
    )
    stop = threading.Event()
    if args.rebuild_every is not None:
        threading.Thread(
            target=rebuild_loop,
            args=(
                args.rebuild_every,
                args.pull,
                build,
                snapshot,
                doc_dir,
                snapshots_dir,
                stop,
            ),
            daemon=True,
        ).start()

    host, port = server.server_address[:2]
    log.info("serving %s at http://%s:%d/%s/", current, host, port, CRATE)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "bound to %s with no auth or TLS; keep it on a trusted network", args.host
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
