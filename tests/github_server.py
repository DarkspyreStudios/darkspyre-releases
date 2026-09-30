"""A local HTTP server that serves release assets the way GitHub does.

GitHub answers `/<owner>/<repo>/releases/download/<tag>/<name>` with a 302 redirect to a signed,
short-lived URL on release-assets.githubusercontent.com. That host answers with
`application/octet-stream`, an ETag, `Accept-Ranges: bytes`, and 206 for a single byte range.

This server reproduces that shape on 127.0.0.1:

- The download path answers 302 to `/release-assets/<id>?sig=<token>`, or 404 for an unknown asset.
- Each signed token serves one request. A reused token answers 403, as an expired signature does,
  so a client that resumes must request the download path again for a fresh redirect.
- The asset path honours one byte range, If-Range against its ETag, and answers 416 for a range
  that starts at or past the end.

Test controls: `truncate_once[name] = n` sends full headers and only n body bytes on the next full
GET of that asset, then drops the connection. `rotate_etag` changes an asset's ETag. `ranges = False`
makes the asset host ignore Range headers. `requests` records every request.
"""
import email.utils
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


def parse_range(header: str, size: int):
    """Return (first, last) for one satisfiable range, None to ignore the header, or "416"."""
    if not header.startswith("bytes=") or "," in header:
        return None
    first_text, _, last_text = header[6:].strip().partition("-")
    try:
        if first_text == "":
            length = int(last_text)
            if length <= 0:
                return "416"
            return max(size - length, 0), size - 1
        first = int(first_text)
        last = int(last_text) if last_text else size - 1
    except ValueError:
        return None
    if first >= size or last < first:
        return "416"
    return first, min(last, size - 1)


class GitHubReleaseServer:
    """Serve {(tag, name): path} for one repository until stop()."""

    def __init__(self, repository: str, assets: dict[tuple[str, str], Path]):
        self.repository = repository
        self.assets = dict(assets)
        self.ids = {key: index + 1 for index, key in enumerate(self.assets)}
        self.tokens: dict[str, tuple[str, str]] = {}
        self.etag_generation: dict[tuple[str, str], int] = {}
        self.truncate_once: dict[str, int] = {}
        self.ranges = True
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                server.handle(self, send_body=True)

            def do_HEAD(self) -> None:
                server.handle(self, send_body=False)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self) -> "GitHubReleaseServer":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def rotate_etag(self, tag: str, name: str) -> None:
        with self.lock:
            self.etag_generation[(tag, name)] = self.etag_generation.get((tag, name), 0) + 1

    @staticmethod
    def empty(handler, status: int, headers: dict | None = None) -> None:
        handler.send_response(status)
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def handle(self, handler, send_body: bool) -> None:
        split = urlsplit(handler.path)
        path = unquote(split.path)
        with self.lock:
            self.requests.append({"method": handler.command, "path": path,
                                  "range": handler.headers.get("Range"), "if_range": handler.headers.get("If-Range")})
        prefix = f"/{self.repository}/releases/download/"
        if path.startswith(prefix):
            tag, _, name = path[len(prefix):].partition("/")
            if (tag, name) not in self.assets:
                return self.empty(handler, 404)
            token = secrets.token_hex(8)
            with self.lock:
                self.tokens[token] = (tag, name)
            location = f"{self.base_url}/release-assets/{self.ids[(tag, name)]}?sig={token}"
            return self.empty(handler, 302, {"Location": location, "Content-Type": "text/html; charset=utf-8"})
        if path.startswith("/release-assets/"):
            token = parse_qs(split.query).get("sig", [""])[0]
            with self.lock:
                key = self.tokens.pop(token, None)
            if key is None or f"/release-assets/{self.ids[key]}" != path:
                return self.empty(handler, 403)
            return self.serve_asset(handler, key, send_body)
        return self.empty(handler, 404)

    def serve_asset(self, handler, key: tuple[str, str], send_body: bool) -> None:
        file = self.assets[key]
        info = file.stat()
        size = info.st_size
        etag = f'"0x{info.st_mtime_ns:X}{size:X}G{self.etag_generation.get(key, 0)}"'
        status, first, last = 200, 0, size - 1
        requested = handler.headers.get("Range") if self.ranges else None
        if_range = handler.headers.get("If-Range")
        if requested and (if_range is None or if_range == etag):
            parsed = parse_range(requested, size)
            if parsed == "416":
                return self.empty(handler, 416, {"Content-Range": f"bytes */{size}"})
            if parsed:
                status, (first, last) = 206, parsed

        cut = None
        if send_body and status == 200:
            with self.lock:
                cut = self.truncate_once.pop(key[1], None)
        handler.send_response(status)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Disposition", f"attachment; filename={key[1]}")
        handler.send_header("Last-Modified", email.utils.formatdate(info.st_mtime, usegmt=True))
        handler.send_header("ETag", etag)
        if self.ranges:
            handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(last - first + 1))
        if status == 206:
            handler.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        handler.end_headers()
        if not send_body:
            return
        remaining = last - first + 1 if cut is None else min(cut, last - first + 1)
        with file.open("rb") as source:
            source.seek(first)
            while remaining > 0:
                block = source.read(min(65536, remaining))
                if not block:
                    break
                try:
                    handler.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(block)
        if cut is not None:
            handler.wfile.flush()
            handler.close_connection = True
