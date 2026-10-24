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
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# 允许 `python -m stage.server` 从仓库根目录跑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.llm import LLM, LLMConfig  # noqa: E402
from orchestrator.orchestrator import OrderRejected, Zhongtang  # noqa: E402
from orchestrator.execute import execute_order  # noqa: E402
from orchestrator.schema import parse_order, validate  # noqa: E402
from stage.cloud import (  # noqa: E402
    CloudError,
    CloudLink,
    config_for,
    fetch_model_ids,
    forget,
    load_links,
    menu_view,
    prepare_link,
    remember,
)

CASE_ROOT = Path(__file__).resolve().parents[2] / "成品"
STAGE_FILE = Path(__file__).resolve().parent / "index.html"
_EXEC_LOCK = threading.Lock()
_CASE_FILE = re.compile(r"^/成品/([^/]+)$")

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


def unknown_model_message(model: str, known: list[str] | None) -> str | None:
    """名单取到了，就不许用名单以外的名字。名单没取到（known 为 None）则不拦。"""
    if not model or known is None:
        return None
    if model in known:
        return None
    have = "、".join(known) if known else "（没有）"
    return f"端点没有模型「{model}」。当前只有：{have}"


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
        path = unquote(urlparse(self.path).path)

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

        case_file = self._case_file(path)
        if case_file is not None:
            kind = "text/html; charset=utf-8" if case_file.suffix.lower() in {".html", ".htm"} else "text/plain; charset=utf-8"
            self._send_bytes(case_file.read_bytes(), kind)
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
        path = urlparse(self.path).path
        if path == "/api/execute":
            self._execute()
            return
        if path == "/api/cloud":
            self._connect_cloud()
            return
        if path == "/api/cloud/forget":
            self._forget_cloud()
            return
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

    def _case_file(self, path: str) -> Path | None:
        """只公开这一场的成品夹，不把各司草稿和案卷外的文件送出去。"""
        match = _CASE_FILE.match(path)
        if not match:
            return None
        name = match.group(1)
        if name in {"", ".", ".."} or "/" in name or "\\" in name:
            return None
        root = CASE_ROOT.resolve()
        target = (root / name).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        if target.is_file():
            return target
        return None

    def _execute(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > 100_000:
            self._send_bytes("请求过大".encode(), "text/plain; charset=utf-8", 413)
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            order = parse_order(json.dumps(data.get("order") or {}, ensure_ascii=False))
            problems = validate(order)
        except Exception as exc:  # noqa: BLE001
            self._send_bytes(
                json.dumps({"ok": False, "error": f"升堂单不合格：{exc}"}, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                400,
            )
            return
        if problems:
            self._send_bytes(
                json.dumps({"ok": False, "error": "；".join(problems)}, ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
                400,
            )
            return

        requirement = str(data.get("requirement") or order.goal)[:2000]
        model = str(data.get("model") or "").strip()
        profile = str(data.get("profile") or "").strip()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            payload = json.dumps(event, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not _EXEC_LOCK.acquire(blocking=False):
            emit({"type": "execute/final", "error": "已有一场在执行，请稍候"})
            return
        try:
            cfg, message = self._call_config(profile, model)
            if message or cfg is None:
                emit({"type": "execute/final", "error": message or "模型没选对"})
                return
            with LLM(cfg) as llm:
                execute_order(llm, order, requirement, CASE_ROOT, on_event=emit)
        except Exception as exc:  # noqa: BLE001
            emit({"type": "execute/final", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            _EXEC_LOCK.release()

    def _list_models(self) -> dict:
        """本机名单给选择器，云端名单另附。云端一项里不带钥匙。"""
        clouds = menu_view(load_links())
        base = CONFIG.resolved_fast_base_url()
        if not base:
            return {"models": [], "current": CONFIG.model, "error": "未配置模型地址", "clouds": clouds}

        req = urllib.request.Request(f"{base}/models", method="GET")
        key = CONFIG.resolved_fast_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"models": [], "current": CONFIG.model, "error": str(exc), "clouds": clouds}

        ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
        return {"models": ids, "current": CONFIG.model, "clouds": clouds}

    def _read_json(self, limit: int = 8000) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length > limit:
            raise CloudError("请求过大")
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CloudError("请求格式不对") from exc
        if not isinstance(data, dict):
            raise CloudError("请求格式不对")
        return data

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def _connect_cloud(self) -> None:
        try:
            data = self._read_json()
            kind, label, base = prepare_link(
                str(data.get("kind") or ""),
                str(data.get("api_key") or ""),
                str(data.get("base_url") or ""),
            )
            key = str(data.get("api_key") or "").strip()
            models = fetch_model_ids(base, key)
        except CloudError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
            return
        remember(CloudLink(kind, label, base, key, models))
        self._send_json({"ok": True, "kind": kind, "label": label, "models": models})

    def _forget_cloud(self) -> None:
        try:
            data = self._read_json()
        except CloudError as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)
            return
        kind = str(data.get("kind") or "").strip()
        if kind not in ("api", "plan"):
            self._send_json({"ok": False, "error": "请选择要断开的那一路"}, 400)
            return
        forget(kind)
        self._send_json({"ok": True, "kind": kind})

    def _call_config(self, profile: str, model: str) -> tuple[LLMConfig | None, str | None]:
        """按所选来源配这一单。本机走启动时的地址，云端走存好的那把钥匙。"""
        profile = (profile or "").strip()
        model = (model or "").strip()
        if profile in ("", "local"):
            if model:
                info = self._list_models()
                known = None if info.get("error") else list(info.get("models") or [])
                message = unknown_model_message(model, known)
                if message:
                    return None, message
            cfg = dataclasses.replace(CONFIG, model=model) if model else CONFIG
            return cfg, None
        link = load_links().get(profile)
        if link is None:
            return None, "这一路云端还没接入"
        if not model:
            return None, "请先选一个云端模型"
        message = unknown_model_message(model, link.models)
        if message:
            return None, message
        return config_for(CONFIG, link, model), None

    def _stream_order(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        requirement = (qs.get("requirement") or [""])[0].strip()
        model = (qs.get("model") or [""])[0].strip()
        profile = (qs.get("profile") or [""])[0].strip()

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

        cfg, message = self._call_config(profile, model)
        if message or cfg is None:
            emit({"type": "order/final", "error": message or "模型没选对"})
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
