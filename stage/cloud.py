"""
升堂 · 云端接入

两路，都走 OpenAI 兼容接口：

  api   自己填 API Key 和地址。地址空着时用 DeepSeek 官方。
  plan  阿里云百炼 Token Plan 订阅。地址写死，钥匙必须以 sk-sp- 开头。

钥匙只落在仓库里的 .shengtang/cloud.json，这个目录不进 Git。
给页面的名单里不带钥匙。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from orchestrator.llm import LLMConfig

PLAN_KIND = "plan"
API_KIND = "api"
PLAN_BASE = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
_KINDS = (API_KIND, PLAN_KIND)
_LOCK = threading.Lock()


class CloudError(ValueError):
    """钥匙或地址不能用，或者名单没拉下来。"""


@dataclass
class CloudLink:
    kind: str
    label: str
    base_url: str
    api_key: str
    models: list[str]


def default_store() -> Path:
    return Path(__file__).resolve().parents[1] / ".shengtang" / "cloud.json"


def prepare_link(kind: str, api_key: str, base_url: str | None) -> tuple[str, str, str]:
    """校验种类、钥匙、地址。Plan 忽略调用方传来的地址。"""
    kind = (kind or "").strip()
    key = (api_key or "").strip()
    if kind not in _KINDS:
        raise CloudError("请选择 API 接口或 Token Plan")
    if not key or any(ch.isspace() for ch in key) or len(key) > 400:
        raise CloudError("请填写一把不含空格的 API Key")
    if kind == PLAN_KIND:
        if not key.startswith("sk-sp-"):
            raise CloudError("Token Plan 的钥匙以 sk-sp- 开头。按量计费的钥匙请走 API 接口")
        return PLAN_KIND, "Token Plan", PLAN_BASE
    return API_KIND, "云端接口", _normalize_base(base_url or "")


def _normalize_base(url: str) -> str:
    text = (url or "").strip() or DEEPSEEK_BASE
    for suffix in ("/chat/completions", "/models"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = text.rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise CloudError("地址要以 http:// 或 https:// 开头")
    if parsed.username or parsed.password:
        raise CloudError("不要把钥匙写进地址里")
    return text


def model_ids_from(data: object) -> list[str]:
    """只认 OpenAI 那种 {data:[{id}]}。去重，保持原顺序。"""
    if not isinstance(data, dict):
        return []
    rows = data.get("data")
    if not isinstance(rows, list):
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ids.append(model_id)
    return ids


def fetch_model_ids(base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    """向对方要模型名单。失败时不把钥匙或对方的响应正文带出去。"""
    req = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudError("钥匙被拒绝了") from exc
        raise CloudError(f"名单没取到（HTTP {exc.code}）") from exc
    except urllib.error.URLError as exc:
        raise CloudError("连不上这个地址") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CloudError("名单不是预期的格式") from exc
    ids = model_ids_from(data)
    if not ids:
        raise CloudError("这个地址没有列出模型")
    return ids


def load_links(path: Path | None = None) -> dict[str, CloudLink]:
    store = path or default_store()
    if not store.is_file():
        return {}
    try:
        raw = json.loads(store.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    links: dict[str, CloudLink] = {}
    for row in raw.get("links") or []:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind not in _KINDS:
            continue
        key = row.get("api_key")
        base = row.get("base_url")
        models = row.get("models")
        if not isinstance(key, str) or not isinstance(base, str) or not isinstance(models, list):
            continue
        ids = [m for m in models if isinstance(m, str) and m.strip()]
        label = "Token Plan" if kind == PLAN_KIND else "云端接口"
        links[kind] = CloudLink(kind, label, base, key, ids)
    return links


def remember(link: CloudLink, path: Path | None = None) -> None:
    store = path or default_store()
    with _LOCK:
        links = load_links(store)
        links[link.kind] = link
        _write(store, links)


def forget(kind: str, path: Path | None = None) -> None:
    store = path or default_store()
    with _LOCK:
        links = load_links(store)
        links.pop(kind, None)
        _write(store, links)


def _write(store: Path, links: dict[str, CloudLink]) -> None:
    store.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "links": [
            {
                "kind": link.kind,
                "label": link.label,
                "base_url": link.base_url,
                "api_key": link.api_key,
                "models": link.models,
            }
            for link in links.values()
        ]
    }
    tmp = store.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(store)


def menu_view(links: dict[str, CloudLink]) -> list[dict]:
    """给页面的名单。没有钥匙。"""
    view = []
    for kind in _KINDS:
        link = links.get(kind)
        if link is None:
            continue
        view.append({"id": link.kind, "label": link.label, "models": list(link.models)})
    return view


def config_for(base: LLMConfig, link: CloudLink, model: str) -> LLMConfig:
    """这一单改走云端。不改原来的本机配置对象。"""
    import dataclasses

    return dataclasses.replace(
        base,
        model=model,
        base_url=link.base_url,
        api_key=link.api_key,
        fast_base_url=link.base_url,
        fast_api_key=link.api_key,
        reasoning=None,
    )
