"""
命令行入口：python -m orchestrator "把升堂工作流做成能看的朝堂场景"

默认连本地 OpenAI 兼容端点。接云端只改 --base-url / --api-key。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import LLM, LLMConfig
from .orchestrator import OrderRejected, Zhongtang
from .schema import describe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m orchestrator",
        description="升堂编排器 · 拆单（带校验兜底）",
    )
    ap.add_argument("requirement", nargs="?", help="主公的需求")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reasoning", default=None, help="本地模型建议 off")
    ap.add_argument("--patch", action="append", default=[], help="profile patch，可重复")
    ap.add_argument("--home", default=None, help="DSH home 目录")
    ap.add_argument("--attempts", type=int, default=3, help="最多重试轮数")
    ap.add_argument("--no-fast", action="store_true", help="拆单也走 agent 回合（慢，但带工具）")
    ap.add_argument("--max-tokens", type=int, default=None, help="裸调输出上限，默认 2000")
    ap.add_argument("--json", action="store_true", help="只输出最终 JSON")
    ap.add_argument("--trace", action="store_true", help="打印每一轮的事件")
    args = ap.parse_args(argv)

    requirement = args.requirement or "把升堂工作流做成一个能看的朝堂场景"
    cfg = LLMConfig.from_env()
    if args.provider:
        cfg.provider = args.provider
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    if args.reasoning:
        cfg.reasoning = args.reasoning
    if args.home:
        cfg.home = args.home
    if args.patch:
        cfg.patches = tuple(str(Path(p).resolve()) for p in args.patch)
    if args.max_tokens:
        cfg.max_tokens = args.max_tokens

    def on_event(ev: dict) -> None:
        if not args.trace:
            return
        kind = ev.get("type")
        if kind == "order/attempt":
            print(f"  · 第 {ev['index']} 轮拆单 …")
        elif kind == "order/rejected":
            print(f"  ✗ 第 {ev['index']} 轮不合格：")
            for p in ev.get("problems", []):
                print(f"      - {p}")
        elif kind == "order/accepted":
            print(f"  ✓ 第 {ev['index']} 轮通过校验")
        elif kind == "order/fast-unavailable":
            print(f"  ! 裸调不可用，退回 agent 回合：{ev.get('reason')}")
        elif kind == "order/failed":
            print("  ✗ 全部轮次失败")

    with LLM(cfg) as llm:
        zt = Zhongtang(
            llm,
            max_attempts=args.attempts,
            on_event=on_event,
            fast=not args.no_fast,
        )
        try:
            order = zt.issue_order(requirement)
        except OrderRejected as exc:
            print("\n拆单失败，应由朝堂显示「中堂请示主公」：", file=sys.stderr)
            for a in exc.attempts:
                print(f"  第 {a.index} 轮：{a.problems}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(order.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(describe(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
