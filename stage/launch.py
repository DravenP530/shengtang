"""
升堂 · 启动

自己找本机上的 OpenAI 兼容模型服务，认到模型名再开朝堂。
网页等端口真正听上了再开，避免先弹出「无法访问」。
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

#: (端口, 路径前缀, 依次尝试的密钥)。空字符串表示不带 Authorization。
CANDIDATES: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (8096, "/v1", ("local", "")),
    (11434, "/v1", ("ollama", "")),
    (1234, "/v1", ("lm-studio", "local", "")),
    (8080, "/v1", ("local", "")),
    (8000, "/v1", ("local", "")),
)

COURT_PORT = 8770


def _model_ids(base: str, key: str) -> list[str]:
    req = urllib.request.Request(base.rstrip("/") + "/models", method="GET")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=1.2) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]


def detect_model() -> tuple[str, str, str] | None:
    """返回 (base_url, model_id, api_key)。认不到就 None。"""
    preferred = (os.environ.get("SHENGTANG_MODEL") or "").strip()
    explicit = (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
    explicit_key = os.environ.get("DEEPSEEK_API_KEY")
    tries: list[tuple[str, tuple[str, ...]]] = []
    if explicit:
        keys = (explicit_key,) if explicit_key is not None else ("local", "")
        tries.append((explicit.rstrip("/"), keys))
    for port, prefix, keys in CANDIDATES:
        tries.append((f"http://127.0.0.1:{port}{prefix}", keys))

    seen: set[str] = set()
    for base, keys in tries:
        if base in seen:
            continue
        seen.add(base)
        for key in keys:
            try:
                ids = _model_ids(base, key)
            except Exception:
                continue
            if not ids:
                continue
            chosen = preferred if preferred in ids else ids[0]
            return base, chosen, key
    return None


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), 0.4):
            return True
    except OSError:
        return False


def _open_when_ready(port: int) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        if _port_open(port):
            webbrowser.open(f"http://127.0.0.1:{port}/")
            return
        time.sleep(0.3)


def main() -> int:
    if _port_open(COURT_PORT):
        print(f"朝堂已经开着： http://127.0.0.1:{COURT_PORT}")
        webbrowser.open(f"http://127.0.0.1:{COURT_PORT}/")
        return 0

    found = detect_model()
    if not found:
        print("没有找到本机模型服务。")
        print("看过这些地址：8096、11434、1234、8080、8000 的 /v1/models")
        print("请先启动本机模型（例如 ninfer、Ollama、LM Studio），再双击一次。")
        return 2

    base, model, key = found
    print(f"已识别模型：{model}")
    print(f"地址：{base}")
    os.environ["DEEPSEEK_BASE_URL"] = base
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["SHENGTANG_MODEL"] = model

    threading.Thread(target=_open_when_ready, args=(COURT_PORT,), daemon=True).start()

    from stage.server import main as serve

    argv = [
        "--host",
        "127.0.0.1",
        "--port",
        str(COURT_PORT),
        "--base-url",
        base,
        "--model",
        model,
    ]
    if key:
        argv.extend(["--api-key", key])
    return serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
