"""Serve the pyrxmesh demo site on port 8001."""

import uvicorn
from pathlib import Path

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
                 "headers": [[b"content-type", ct.encode()]]})
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    print(f"Serving {SITE_DIR} at http://localhost:8001")
    uvicorn.run("serve_demo:app", host="0.0.0.0", port=8001)
