"""
升堂 · 模型接入层

两条路，按任务性质选：

  quick()  裸调 OpenAI 兼容接口。不走 agent loop，不加载工具，不起 runtime。
           适合「读一段话、吐一段 JSON」这种纯生成任务——拆单、改写、翻译。
           实测比走 SDK 快一到两个数量级。

  chat()   走 DeepSeek Harness SDK。带工具、带会话、带文件读写。
           适合真正要干活的阶段——执行、勘验。

runtime 是懒启动的：只用 quick() 的话，一个子进程都不会起。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

#: 官方 provider 对应的裸调地址，仅在没显式给 base_url 时兜底。
DEFAULT_BASE_URLS = {
    "deepseek-official": "https://api.deepseek.com/v1",
}


class FastPathUnavailable(RuntimeError):
    """没有可用于裸调的地址或凭据。"""


@dataclass
class LLMConfig:
    """模型连接参数。"""

    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    #: 裸调用的地址与凭据；留空则回退到 base_url / api_key
    fast_base_url: str | None = None
    fast_api_key: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    reasoning: str | None = None
    #: 裸调的默认输出上限。拆单这种任务 2000 足够，别给 256k。
    max_tokens: int = 2000
    temperature: float = 0.2
    timeout: float = 180.0
    home: str = "~/.shengtang/dsh-home"
    cwd: str | None = None
    patches: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.environ.get("SHENGTANG_PROVIDER", "deepseek-official"),
            model=os.environ.get("SHENGTANG_MODEL", "deepseek-v4-flash"),
            fast_base_url=os.environ.get("SHENGTANG_FAST_BASE_URL") or None,
            fast_api_key=os.environ.get("SHENGTANG_FAST_API_KEY") or None,
            base_url=os.environ.get("DEEPSEEK_BASE_URL") or None,
            api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
            reasoning=os.environ.get("SHENGTANG_REASONING") or None,
        )

    # ------------------------------------------------------------ 解析

    def resolved_fast_base_url(self) -> str | None:
        url = self.fast_base_url or self.base_url
        if url:
            return url.rstrip("/")
        return DEFAULT_BASE_URLS.get(self.provider)

    def resolved_fast_api_key(self) -> str | None:
        return self.fast_api_key or self.api_key


class LLM:
    """一个 LLM 实例同时具备快慢两条路。runtime 按需启动。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._harness: DeepSeekHarness | None = None
        self._home = Path(self.config.home).expanduser().resolve()
        self._home.mkdir(parents=True, exist_ok=True)
        self._cwd = Path(self.config.cwd).resolve() if self.config.cwd else self._home
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._calls = 0

    # ------------------------------------------------------------ 生命周期

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._harness is not None:
            self._harness.__exit__(None, None, None)
            self._harness = None

    def _ensure_harness(self) -> DeepSeekHarness:
        """第一次需要 agent 能力时才拉起 runtime。"""
        if self._harness is None:
            cfg = DeepSeekHarnessConfig(
                provider=self.config.provider,
                model=self.config.model,
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                reasoning_effort=self.config.reasoning,
                dsh_home=str(self._home),
                cwd=str(self._cwd),
                patches=self.config.patches,
            )
            self._harness = DeepSeekHarness(cfg)
            self._harness.__enter__()
        return self._harness

    # ------------------------------------------------------------ 快路

    def quick(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """裸调 OpenAI 兼容 /chat/completions。不起 runtime、不带工具。"""
        base_url = self.config.resolved_fast_base_url()
        if not base_url:
            raise FastPathUnavailable(
                "没有可用的裸调地址。请设置 fast_base_url 或环境变量 "
                "DEEPSEEK_BASE_URL（例如 http://127.0.0.1:8096/v1）"
            )

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        api_key = self.config.resolved_fast_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"裸调失败 HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"裸调连不上 {base_url}: {exc.reason}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"裸调返回结构异常: {str(data)[:400]}") from exc

        self._calls += 1
        return content or ""

    # ------------------------------------------------------------ 慢路

    def chat(self, prompt: str) -> str:
        """走 SDK 的完整 agent 回合。带工具与会话，适合要动手的活。"""
        harness = self._ensure_harness()
        self._calls += 1
        session_id = f"shengtang-{time.strftime('%Y%m%d-%H%M%S')}-{self._calls}"
        result = harness.run(prompt, session_id=session_id)

        if result.finish_reason not in (None, "completed", "stop"):
            raise RuntimeError(
                f"模型调用未完成：{result.finish_reason}；"
                f"最后事件 {result.events[-1] if result.events else '无'}"
            )

        return result.final_response or ""
