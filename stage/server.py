"""
升堂 · 演出服务

把编排器接到朝堂页面上。做三件事：
  1. 托管朝堂页面（stage/index.html）
  2. 提供 /api/order 事件流：拆单过程中每一步都实时推给前端
  3. 把模型的失败如实转成「请示主公」，不假装成功

跑法：
  python -m stage.server --model qwen3.8-27b ^
      --base-url http://127.0.0.1:8096/v1 --api-key local

然后浏览器打开 http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# 允许 `python -m stage.server` 从仓库根目录跑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.llm import LLM, LLMConfig  # noqa: E402
from orchestrator.orchestrator import OrderRejected, Zhongtang  # noqa: E402

STAGE_FILE = Path(__file__).resolve().parent / "index.html"

CONFIG = LLMConfig.from_env()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # 关掉默认的逐请求日志，只保留错误
    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        if str(args[1] if len(args) > 1 else "")[:1] in ("4", "5"):
            sys.stderr.write("[stage] " + (fmt % args) + "\n")

    # ------------------------------------------------------------ 页面

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            if not STAGE_FILE.exists():
                self._send_bytes("stage/index.html 不存在".encode(), "text/plain; charset=utf-8", 500)
                return
            self._send_bytes(STAGE_FILE.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/order":
            self._stream_order()
            return

        if path == "/api/health":
            self._send_bytes(
                json.dumps({"ok": True, "model": CONFIG.model}).encode(),
                "application/json; charset=utf-8",
            )
            return

        self._send_bytes("not found".encode(), "text/plain; charset=utf-8", 404)

    # ------------------------------------------------------------ 事件流

    def _stream_order(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        requirement = (qs.get("requirement") or [""])[0].strip()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            try:
                payload = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        if not requirement:
            emit({"type": "order/final", "error": "需求为空"})
            return

        try:
            with LLM(CONFIG) as llm:
                zt = Zhongtang(llm, max_attempts=3, on_event=emit, fast=True)
                order = zt.issue_order(requirement)
                emit({"type": "order/final", "order": order.to_dict()})
        except OrderRejected as exc:
            # 终局兜底：如实告诉前端「拆不出来」，由朝堂显示「请示主公」
            emit(
                {
                    "type": "order/final",
                    "error": str(exc),
                    "attempts": [
                        {"index": a.index, "problems": a.problems} for a in exc.attempts
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001
            emit({"type": "order/final", "error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stage.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args(argv)

    for attr in ("provider", "model", "base_url", "api_key", "max_tokens"):
        v = getattr(args, attr)
        if v:
            setattr(CONFIG, attr, v)

    if not CONFIG.resolved_fast_base_url():
        print("缺少模型地址：请加 --base-url 或设置 DEEPSEEK_BASE_URL", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[stage] 朝堂已开： http://{args.host}:{args.port}")
    print(f"[stage] 模型：{CONFIG.model}  ←  {CONFIG.resolved_fast_base_url()}")
    print("[stage] Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stage] 已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
