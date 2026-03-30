"""Serve the pyrxmesh demo site on port 8001."""

import os
import signal
import subprocess
import uvicorn
from pathlib import Path


def kill_existing():
    """Kill any existing uvicorn processes serving on port 8001."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", ":8001"], capture_output=True, text=True)
        for pid in result.stdout.strip().split():
            pid = int(pid)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                print(f"Killed existing server (PID {pid})")
    except Exception:
        pass

SITE_DIR = Path(__file__).parent / "docs" / "_site"


async def app(scope, receive, send):
    if scope["type"] != "http":
        return

    path = scope["path"].lstrip("/") or "index.html"
    file = SITE_DIR / path

    if not file.is_file() or not file.resolve().is_relative_to(SITE_DIR.resolve()):
        await send({"type": "http.response.start", "status": 404,
                     "headers": [[b"content-type", b"text/plain"]]})
        await send({"type": "http.response.body", "body": b"Not found"})
        return

    content_types = {
        ".html": "text/html", ".png": "image/png", ".jpg": "image/jpeg",
        ".css": "text/css", ".js": "application/javascript",
    }
    ct = content_types.get(file.suffix, "application/octet-stream")
    body = file.read_bytes()

    await send({"type": "http.response.start", "status": 200,
                 "headers": [[b"content-type", ct.encode()],
                              [b"cache-control", b"no-cache, no-store, must-revalidate"],
                              [b"pragma", b"no-cache"],
                              [b"expires", b"0"]]})
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    kill_existing()
    # Install tabbed index
    import shutil
    tab_src = Path(__file__).parent / "docs" / "_site_index.html"
    if tab_src.exists():
        SITE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(tab_src, SITE_DIR / "index.html")
        print(f"Installed tabbed index")
    print(f"Serving {SITE_DIR} at http://localhost:8001")
    uvicorn.run("serve_demo:app", host="0.0.0.0", port=8001)
