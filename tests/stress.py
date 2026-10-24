"""
升堂 · 真实模型压力测试

同一需求连跑 N 次，统计「第一轮就合格」的比例。
这个数字比任何 benchmark 都有用：它决定本地模型能不能直接驱动编排器。

跑法：
  python tests/stress.py --n 6 --provider local-ninfer --model qwen3.8-27b ^
      --patch ../工具/local-ninfer.patch.yml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.llm import LLM, LLMConfig  # noqa: E402
from orchestrator.orchestrator import OrderRejected, Zhongtang  # noqa: E402

REQUIREMENT = (
    "把升堂工作流做成一个能看的朝堂场景：输入需求点升堂，六司齐呼，"
    "中堂拆单，刑部勘验后汇总。验收标准：能完整跑完一轮并结案。"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--patch", action="append", default=[])
    ap.add_argument("--home", default=None)
    args = ap.parse_args()

    cfg = LLMConfig.from_env()
    for attr in ("provider", "model", "base_url", "api_key", "reasoning", "home"):
        v = getattr(args, attr)
        if v:
            setattr(cfg, attr, v)
    if args.patch:
        cfg.patches = tuple(str(Path(p).resolve()) for p in args.patch)

    first_pass = retried = failed = 0
    rounds: list[int] = []
    reasons: dict[str, int] = {}
    t0 = time.time()

    with LLM(cfg) as llm:
        for i in range(1, args.n + 1):
            zt = Zhongtang(llm, max_attempts=args.attempts)
            try:
                zt.issue_order(REQUIREMENT)
                used = len(zt.attempts)
                rounds.append(used)
                if used == 1:
                    first_pass += 1
                else:
                    retried += 1
                mark = "一次过" if used == 1 else f"第 {used} 轮才过"
            except OrderRejected as exc:
                failed += 1
                rounds.append(args.attempts)
                mark = "失败"
                for a in exc.attempts:
                    for p in a.problems:
                        key = p.split("，")[0][:40]
                        reasons[key] = reasons.get(key, 0) + 1
            print(f"  第 {i:>2} 次：{mark}", flush=True)

    n = args.n
    print()
    print(f"样本 {n} 次，耗时 {time.time() - t0:.0f} 秒")
    print(f"  一次过   {first_pass}  ({first_pass / n:.0%})")
    print(f"  重试后过 {retried}  ({retried / n:.0%})")
    print(f"  失败     {failed}  ({failed / n:.0%})")
    print(f"  平均轮数 {sum(rounds) / len(rounds):.2f}")

    if reasons:
        print()
        print("失败单子里的问题分布：")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>2} 次  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
