"""
升堂 · 演出服务

把编排器接到朝堂页面上。做四件事：
  1. 托管朝堂页面（stage/index.html）
  2. 提供 /api/order 事件流：拆单过程中每一步都实时推给前端
  3. 把模型的失败如实转成「请示主公」，不假装成功
  4. 提供 /api/answer：中堂问话时，主公的回答经这条 POST 送回，后端才继续拆

跑法：
  python -m stage.server --model qwen3.8-27b ^
      --base-url http://127.0.0.1:8096/v1 --api-key local

然后浏览器打开 http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
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

#: 同一时间只允许一张单在拆。浏览器断开重连时 EventSource 会自动重发，
#: 没有这道闸就会叠出好几个模型调用，把本地模型压死。
_ORDER_LOCK = threading.Lock()

#: 中堂问话的中转站：sid -> {answer, done, gone}。
#: SSE 是单向流（只能服务端推），主公的回答必须靠另一条 POST 通道送回。
_PENDING: dict[str, dict] = {}
_PENDING_LOCK = threading.Lock()

#: 等主公回答的上限（秒）。超时就按原需求继续拆——问话是增益，不是门槛。
ANSWER_TIMEOUT = 180.0


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # noqa: D102
        exc = sys.exc_info()[1]
        # 浏览器预连接后立刻断开是常态，不必刷一屏堆栈
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


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

        if path == "/api/models":
            self._send_bytes(
                json.dumps(self._list_models(), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
            return

        if path == "/favicon.ico":
            self._send_bytes(b"", "image/x-icon", 204)
            return

        self._send_bytes("not found".encode(), "text/plain; charset=utf-8", 404)

    def do_HEAD(self) -> None:  # noqa: N802
        """预览面板/健康检查会用 HEAD 探活。不实现的话返回 501，页面会被判成打不开。"""
        path = urlparse(self.path).path
        if path in ("/", "/index.html") and STAGE_FILE.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(STAGE_FILE.stat().st_size))
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        """主公的回话通道。SSE 只能服务端推客户端，回答得靠这条 POST 送回来。"""
        path = urlparse(self.path).path
        if path != "/api/answer":
            self._send_bytes("not found".encode(), "text/plain; charset=utf-8", 404)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""

        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            self._send_bytes(
                json.dumps({"ok": False, "error": "请求体不是 JSON"}).encode(),
                "application/json; charset=utf-8",
                400,
            )
            return

        sid = str(data.get("sid") or "")
        answer = str(data.get("answer") or "")[:2000]
        with _PENDING_LOCK:
            box = _PENDING.get(sid)

        if box is None:
            self._send_bytes(
                json.dumps({"ok": False, "error": "没有在等这个问题"}).encode(),
                "application/json; charset=utf-8",
                404,
            )
            return

        box["answer"] = answer
        box["done"].set()
        self._send_bytes(json.dumps({"ok": True}).encode(), "application/json; charset=utf-8")

    # ------------------------------------------------------------ 事件流

    def _list_models(self) -> dict:
        """把端点的 /models 转给前端，供模型选择器用。"""
        base = CONFIG.resolved_fast_base_url()
        if not base:
            return {"models": [], "current": CONFIG.model, "error": "未配置模型地址"}

        req = urllib.request.Request(f"{base}/models", method="GET")
        key = CONFIG.resolved_fast_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"models": [], "current": CONFIG.model, "error": str(exc)}

        ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        return {"models": ids, "current": CONFIG.model}

    def _stream_order(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        requirement = (qs.get("requirement") or [""])[0].strip()
        model = (qs.get("model") or [""])[0].strip()

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

        # 同一时间只拆一张单。EventSource 断线会自动重发，不挡就会叠出多个模型调用。
        if not _ORDER_LOCK.acquire(blocking=False):
            emit({"type": "order/final", "error": "已有单在拆，请稍候再点"})
            return

        def clarifier(question: str) -> str:
            """中堂问话：把问题推给前端，然后停下来等主公回答。

            等待期间每秒发一个 SSE 注释心跳——既让连接活着，
            也能在主公关掉页面时立刻发现（往死连接上写就会炸）。
            """
            sid = uuid.uuid4().hex[:8]
            box = {"answer": "", "done": threading.Event(), "gone": False}
            with _PENDING_LOCK:
                _PENDING[sid] = box
            try:
                emit({"type": "order/question", "sid": sid, "question": question})
                deadline = time.monotonic() + ANSWER_TIMEOUT
                while not box["done"].wait(1.0):
                    if time.monotonic() >= deadline:
                        break
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except OSError:
                        box["gone"] = True
                        break
                if box["gone"]:
                    raise ConnectionError("主公已离场，问话作废")
                return box["answer"]
            finally:
                with _PENDING_LOCK:
                    _PENDING.pop(sid, None)

        try:
            # 每张单可以带自己的模型，互不影响
            cfg = dataclasses.replace(CONFIG, model=model) if model else CONFIG
            with LLM(cfg) as llm:
                zt = Zhongtang(llm, max_attempts=3, on_event=emit, fast=True)
                order = zt.issue_order(requirement, clarifier=clarifier)
                emit({"type": "order/final", "order": order.to_dict(), "model": cfg.model})
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
        finally:
            _ORDER_LOCK.release()


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

    server = Server((args.host, args.port), Handler)
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
