#!/usr/bin/env python3
"""Tiny web monitor for BONES-SEED sync status."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


def human_tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return "log not created yet"
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def sanitize_log(text: str) -> str:
    text = re.sub(r"https://cas-bridge\.xethub\.hf\.co/\S+", "https://cas-bridge.xethub.hf.co/[signed-url-redacted]", text)
    text = re.sub(r"(Authorization:\s*Bearer\s+)\S+", r"\1[redacted]", text)
    return text


def bytes_gb(n: int | float) -> float:
    return round(float(n) / 1_000_000_000, 3)


def allocated_size(path: Path) -> int:
    """Return actual disk bytes, not sparse-file apparent size."""
    st = path.stat()
    return int(st.st_blocks) * 512


def cache_download_progress(status: dict[str, object]) -> dict[str, object]:
    data_dir = Path(str(status.get("data_dir", "")))
    current = None
    archives = status.get("archives", {})
    if isinstance(archives, dict):
        for name, item in archives.items():
            if isinstance(item, dict) and item.get("state") == "downloading":
                current = (str(name), item)
                break
    if current is None or not data_dir.exists():
        return {}

    name, item = current
    archive_path = data_dir / name
    aria_path = data_dir / f"{name}.aria2"
    cache = data_dir / ".cache" / "huggingface" / "download"
    incomplete_files = sorted(cache.glob("*.incomplete"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if archive_path.exists():
        size = allocated_size(archive_path)
        active_temp_files = [archive_path.name] + ([aria_path.name] if aria_path.exists() else [])
    else:
        size = sum(p.stat().st_size for p in incomplete_files if p.exists())
        active_temp_files = [p.name for p in incomplete_files[:3]]
    expected_gb = float(item.get("expected_size_gb", 0) or 0)
    pct = round((bytes_gb(size) / expected_gb) * 100.0, 2) if expected_gb > 0 else None
    return {
        "archive": name,
        "downloaded_gb": bytes_gb(size),
        "expected_gb": expected_gb,
        "percent": pct,
        "active_temp_files": active_temp_files,
    }


def render(status_path: Path, log_path: Path) -> bytes:
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {"state": "missing", "message": "status file not created yet"}
    state = str(status.get("state", "unknown"))
    disk = status.get("disk", {})
    archives = status.get("archives", {})
    progress = cache_download_progress(status)
    rows = []
    for name, item in archives.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(item.get('state', '')))}</td>"
            f"<td>{html.escape(str(item.get('size_gb', item.get('expected_size_gb', ''))))}</td>"
            f"<td>{html.escape(str(item.get('archive_deleted', False)))}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows) or "<tr><td colspan='4'>waiting</td></tr>"
    log = html.escape(sanitize_log(human_tail(log_path)))
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="8">
<title>BONES-SEED Sync</title>
<style>
body {{ margin:0; background:#050606; color:#f3efe8; font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:18px; }}
.pill {{ display:inline-block; padding:5px 10px; border:1px solid #ff8a18; color:#ff8a18; border-radius:999px; font-weight:700; }}
.grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:16px 0; }}
.card {{ border:1px solid #2b302d; border-radius:8px; background:#101312; padding:12px; }}
.label {{ color:#9aa29b; font-size:12px; }}
.value {{ font-size:20px; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
td,th {{ border-bottom:1px solid #2b302d; padding:8px; text-align:left; font-size:13px; }}
pre {{ white-space:pre-wrap; background:#080a09; border:1px solid #2b302d; border-radius:8px; padding:12px; font-size:12px; overflow-wrap:anywhere; }}
</style></head>
<body><main>
<h1>BONES-SEED Sync</h1>
<span class="pill">{html.escape(state)}</span>
<p>{html.escape(str(status.get("message", "")))}</p>
<div class="grid">
  <div class="card"><div class="label">Current stage</div><div class="value">{html.escape(str(status.get("current_stage", "")))}</div></div>
  <div class="card"><div class="label">Extracted size</div><div class="value">{html.escape(str(status.get("extracted_size_gb", 0)))} GB</div></div>
  <div class="card"><div class="label">Disk free</div><div class="value">{html.escape(str(disk.get("free_gb", "")))} GB</div></div>
  <div class="card"><div class="label">Active download</div><div class="value">{html.escape(str(progress.get("archive", "")))}</div></div>
  <div class="card"><div class="label">Downloaded</div><div class="value">{html.escape(str(progress.get("downloaded_gb", "")))} / {html.escape(str(progress.get("expected_gb", "")))} GB</div></div>
  <div class="card"><div class="label">Progress</div><div class="value">{html.escape(str(progress.get("percent", "")))}%</div></div>
</div>
<table><thead><tr><th>Archive</th><th>Status</th><th>GB</th><th>Deleted tar</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Log Tail</h2><pre>{log}</pre>
</main></body></html>"""
    return body.encode()


class Handler(BaseHTTPRequestHandler):
    status_path: Path
    log_path: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/status.json":
            body = self.status_path.read_bytes() if self.status_path.exists() else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/log":
            if self.log_path.exists():
                body = sanitize_log(self.log_path.read_text(errors="replace")).encode()
            else:
                body = b""
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        body = render(self.status_path, self.log_path)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[boneseed-monitor] {self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8822)
    parser.add_argument("--status", type=Path, default=Path("alpha_motion2/wbc_expert/runs/boneseed_sync_status.json"))
    parser.add_argument("--log", type=Path, default=Path("alpha_motion2/wbc_expert/runs/boneseed_sync.log"))
    args = parser.parse_args()
    Handler.status_path = args.status.expanduser().resolve()
    Handler.log_path = args.log.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[OK] BONES-SEED monitor listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
