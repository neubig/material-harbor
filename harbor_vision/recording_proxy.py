"""Recording pass-through proxy that proves whether image blocks reach the API.

Purpose
-------
A correct answer on a pixel-only task is strong evidence that vision worked, but
it is still indirect: the model could in principle have been told the answer.
This proxy removes that doubt by capturing the *actual* JSON body the SDK sent.

It sits between the agent container and the real endpoint, forwards requests
unchanged, and writes a sanitized record per request containing only structural
facts -- no prompts, no base64 payloads, no credentials:

* whether ``content`` is a list or a string,
* the block ``type`` values it contained,
* how many image blocks there were,
* the model name and a hash of the body,
* whether the image data survived intact (length + hash of the base64 blob).

The blob hash is what makes this airtight: it can be compared against the hash
of the figure actually placed in the task, so "the pixels arrived" becomes a
byte-level claim rather than an inference from a score.

Running it
----------
    python harbor_vision/recording_proxy.py --port 18110 \
        --upstream "$LLM_BASE_URL" --log-dir <dir> --record-requests

Bind on 0.0.0.0 so the Docker bridge can reach it. The task then points
``LLM_BASE_URL`` at ``http://<host-gateway>:18110``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Bodies are large (base64 images); keep a generous but bounded read.
MAX_BODY = 64 * 1024 * 1024


def summarize_content(content) -> dict:
    """Describe content structure without retaining any of its text or pixels."""
    if isinstance(content, str):
        return {"kind": "string", "length": len(content)}
    if not isinstance(content, list):
        return {"kind": type(content).__name__}

    blocks = []
    images = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": type(block).__name__})
            continue
        block_type = block.get("type")
        entry = {"type": block_type}
        if block_type == "text":
            entry["length"] = len(block.get("text") or "")
        elif block_type == "image_url":
            url = (block.get("image_url") or {}).get("url") or ""
            entry["scheme"] = url.split(",", 1)[0].split(";", 1)[0]
            if url.startswith("data:") and "," in url:
                payload = url.split(",", 1)[1]
                try:
                    raw = base64.b64decode(payload, validate=True)
                    entry["bytes"] = len(raw)
                    entry["sha256"] = hashlib.sha256(raw).hexdigest()
                except Exception:
                    entry["bytes"] = None
                    entry["sha256"] = None
            images.append(entry)
        blocks.append(entry)

    return {
        "kind": "list",
        "blocks": blocks,
        "image_block_count": len(images),
        "images": images,
    }


def summarize(body: dict) -> dict:
    messages = body.get("messages") or []
    return {
        "model": body.get("model"),
        "message_count": len(messages),
        "messages": [
            {
                "index": i,
                "role": m.get("role"),
                "has_tool_calls": bool(m.get("tool_calls")),
                "content": summarize_content(m.get("content")),
            }
            for i, m in enumerate(messages)
            if isinstance(m, dict)
        ],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ""
    log_dir: Path | None = None
    record = False
    lock = threading.Lock()
    counter = 0

    def log_message(self, *args) -> None:  # keep stderr quiet
        pass

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        record = None
        if self.record and raw:
            try:
                body = json.loads(raw)
                if isinstance(body, dict) and "messages" in body:
                    record = summarize(body)
                    record["body_sha256"] = hashlib.sha256(raw).hexdigest()
                    record["body_bytes"] = len(raw)
            except Exception as exc:
                record = {"error": f"{type(exc).__name__}: {exc}"}

        if record is not None and self.log_dir is not None:
            with Handler.lock:
                Handler.counter += 1
                index = Handler.counter
            record["sequence"] = index
            record["timestamp"] = time.time()
            (self.log_dir / f"request-{index:04d}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )

        target = self.upstream.rstrip("/") + self.path
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "accept-encoding")
        }
        request = urllib.request.Request(
            target, data=raw or None, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
                status = response.status
                out_headers = response.headers
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            out_headers = exc.headers
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode()
            status = 502
            out_headers = {}

        self.send_response(status)
        for key, value in out_headers.items():
            if key.lower() in ("content-length", "transfer-encoding", "connection"):
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        self._forward("POST")

    def do_GET(self) -> None:
        self._forward("GET")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--record-requests", action="store_true")
    parser.add_argument("--seconds", type=float, default=0, help="0 = run until killed")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    Handler.upstream = args.upstream
    Handler.log_dir = args.log_dir
    Handler.record = args.record_requests

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"proxy on 0.0.0.0:{args.port} -> {args.upstream}", flush=True)
    if args.seconds:
        threading.Timer(args.seconds, server.shutdown).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
