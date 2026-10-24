"""
升堂 · 兜底测试

不起模型、不联网。用一个假的 LLM 按脚本返回内容，
专门验证「模型输出不合格时，编排器能不能拦住」。

跑法：python tests/test_orchestrator.py
（不依赖 pytest，纯断言，clone 下来就能跑）
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.llm import FastPathUnavailable  # noqa: E402
from orchestrator.orchestrator import OrderRejected, Zhongtang, parse_question  # noqa: E402
from orchestrator.schema import ParseError, parse_order, validate  # noqa: E402


# ---------------------------------------------------------------- 测试替身


class ScriptedLLM:
    """按顺序吐预设内容，模拟模型的不同表现。快慢两条路都记账。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.fast_calls = 0
        self.chat_calls = 0
        self.prompts: list[str] = []

    def _next(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls > len(self.responses):
            raise AssertionError("测试脚本里的响应不够用，说明重试次数超出预期")
        return self.responses[self.calls - 1]

    def quick(self, prompt: str) -> str:
        self.fast_calls += 1
        return self._next(prompt)

    def chat(self, prompt: str) -> str:
        self.chat_calls += 1
        return self._next(prompt)


GOOD = """{"目标":"做朝堂场景","任务":[
{"司":"礼部","做什么":"写文案","交付物":"script.txt"},
{"司":"兵部","做什么":"写页面","交付物":"index.html"},
{"司":"工部","做什么":"打包","交付物":"package/"}]}"""


def problems_of(text: str) -> list[str]:
    return validate(parse_order(text))


# ---------------------------------------------------------------- 解析兜底


def test_parse_tolerates_code_fence():
    """模型爱用代码块包 JSON，解析层必须忍。"""
    fenced = "好的，这是结果：\n```json\n" + GOOD + "\n```\n以上。"
    order = parse_order(fenced)
    assert order.goal == "做朝堂场景"
    assert len(order.tasks) == 3


def test_parse_rejects_garbage():
    """连 JSON 都不是 → 抛 ParseError，由上层当「不合格」处理。"""
    try:
        parse_order("我觉得这个需求可以做，分三步走。")
    except ParseError:
        return
    raise AssertionError("非 JSON 输出应当抛 ParseError")


# ---------------------------------------------------------------- 校验兜底


def test_reject_auditor_in_tasks():
    """刑部只审不做——派给刑部是最严重的违规，必须拦住。"""
    bad = """{"目标":"做场景","任务":[
    {"司":"刑部","做什么":"核验","交付物":"报告"},
    {"司":"兵部","做什么":"写页面","交付物":"index.html"},
    {"司":"礼部","做什么":"写文案","交付物":"script.txt"}]}"""
    problems = problems_of(bad)
    assert any("刑部" in p for p in problems), problems


def test_reject_missing_deliverable():
    bad = """{"目标":"做场景","任务":[
    {"司":"兵部","做什么":"写页面","交付物":""},
    {"司":"礼部","做什么":"写文案","交付物":"script.txt"},
    {"司":"工部","做什么":"打包","交付物":"package/"}]}"""
    problems = problems_of(bad)
    assert any("交付物" in p for p in problems), problems


def test_reject_vague_deliverable():
    bad = """{"目标":"做场景","任务":[
    {"司":"兵部","做什么":"写页面","交付物":"研究一下"},
    {"司":"礼部","做什么":"写文案","交付物":"script.txt"},
    {"司":"工部","做什么":"打包","交付物":"package/"}]}"""
    problems = problems_of(bad)
    assert any("空泛" in p for p in problems), problems


def test_reject_duplicate_dept():
    """同一个司塞 3 条 → 分配不合理，退回。"""
    bad = """{"目标":"做场景","任务":[
    {"司":"兵部","做什么":"a","交付物":"a.txt"},
    {"司":"兵部","做什么":"b","交付物":"b.txt"},
    {"司":"兵部","做什么":"c","交付物":"c.txt"}]}"""
    problems = problems_of(bad)
    assert any("兵部" in p for p in problems), problems


def test_reject_too_many_tasks():
    tasks = ",".join(
        f'{{"司":"{d}","做什么":"x","交付物":"{d}.txt"}}'
        for d in ["吏部", "户部", "礼部", "兵部", "工部", "吏部"]
    )
    problems = problems_of(f'{{"目标":"做场景","任务":[{tasks}]}}')
    assert any("超过上限" in p for p in problems), problems


def test_accept_good_order():
    assert problems_of(GOOD) == []


# ---------------------------------------------------------------- 重试兜底


def test_retry_then_success():
    """第一次派活给刑部，第二次改对了 → 应当返回结果，且用满 2 轮。"""
    bad = """{"目标":"做场景","任务":[
    {"司":"刑部","做什么":"核验","交付物":"报告"},
    {"司":"兵部","做什么":"写页面","交付物":"index.html"},
    {"司":"礼部","做什么":"写文案","交付物":"script.txt"}]}"""
    llm = ScriptedLLM([bad, GOOD])
    events: list[dict] = []
    zt = Zhongtang(llm, max_attempts=3, on_event=events.append)

    order = zt.issue_order("做朝堂场景")

    assert order.goal == "做朝堂场景"
    assert llm.calls == 2, f"应当在第 2 轮成功，实际调用 {llm.calls} 次"
    assert any(e["type"] == "order/rejected" for e in events)
    assert any(e["type"] == "order/accepted" for e in events)
    # 第二轮提示词里必须带着上一轮的问题，否则模型不知道改什么
    assert "刑部" in llm.prompts[1]


def test_all_attempts_fail_raises():
    """三轮都不合格 → 必须抛错，不能把坏单子放出去。"""
    bad = """{"目标":"","任务":[]}"""
    llm = ScriptedLLM([bad, bad, bad])
    zt = Zhongtang(llm, max_attempts=3)

    try:
        zt.issue_order("做朝堂场景")
    except OrderRejected as exc:
        assert len(exc.attempts) == 3
        return
    raise AssertionError("全部轮次失败时必须抛 OrderRejected")


def test_model_crash_counts_as_failure():
    """模型层抛异常（比如没配 key）→ 当成这一轮失败，继续重试，而不是崩掉。"""

    class BoomLLM:
        def __init__(self) -> None:
            self.calls = 0

        def quick(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("MISSING_CREDENTIAL")
            return GOOD

        def chat(self, prompt: str) -> str:  # pragma: no cover - 不该走到
            raise AssertionError("快路可用时不应回退到 chat")

    llm = BoomLLM()
    zt = Zhongtang(llm, max_attempts=3)
    order = zt.issue_order("做朝堂场景")
    assert order.goal == "做朝堂场景"
    assert llm.calls == 2


def test_uses_fast_path_by_default():
    """拆单是纯生成任务，默认应当走裸调，不起 runtime。"""
    llm = ScriptedLLM([GOOD])
    Zhongtang(llm, max_attempts=1).issue_order("做朝堂场景")
    assert llm.fast_calls == 1
    assert llm.chat_calls == 0


def test_falls_back_to_chat_when_fast_unavailable():
    """没配裸调地址时，退回 agent 回合，而不是直接失败。"""

    class NoFastLLM:
        def __init__(self) -> None:
            self.chat_calls = 0
            self.events: list[dict] = []

        def quick(self, prompt: str) -> str:
            raise FastPathUnavailable("没有地址")

        def chat(self, prompt: str) -> str:
            self.chat_calls += 1
            return GOOD

    llm = NoFastLLM()
    events: list[dict] = []
    order = Zhongtang(llm, max_attempts=1, on_event=events.append).issue_order("做朝堂场景")

    assert order.goal == "做朝堂场景"
    assert llm.chat_calls == 1
    assert any(e["type"] == "order/fast-unavailable" for e in events)


def test_no_fast_flag_forces_chat():
    llm = ScriptedLLM([GOOD])
    Zhongtang(llm, max_attempts=1, fast=False).issue_order("做朝堂场景")
    assert llm.chat_calls == 1
    assert llm.fast_calls == 0


def test_empty_requirement_rejected():
    zt = Zhongtang(ScriptedLLM([]), max_attempts=1)
    try:
        zt.issue_order("   ")
    except ValueError:
        return
    raise AssertionError("空需求应当直接拒绝，不该浪费一次模型调用")


# ---------------------------------------------------------------- 澄清兜底（第 0 层）

def test_parse_question():
    """只认「一句真问题」。无疑问、跑题、长文、JSON 一律当没问。"""
    assert parse_question("要做到什么程度算完？") == "要做到什么程度算完？"
    assert parse_question("  「要不要出图？」  ") == "要不要出图？"
    assert parse_question("1. 交付物是什么？") == "交付物是什么？"
    assert parse_question("无疑问") is None
    assert parse_question("无疑问，需求很清楚。") is None
    assert parse_question('{"目标":"x","任务":[]}') is None
    assert parse_question("```json\n{\"目标\":\"x\"}\n```") is None
    assert parse_question("") is None
    assert parse_question("问" * 200) is None


def test_clarifier_answer_feeds_order_prompt():
    """问了就要用上：回答必须回灌进拆单提示词，否则等于白问。"""
    llm = ScriptedLLM(["验收标准是什么？", GOOD])
    seen: dict[str, str] = {}
    events: list[dict] = []

    def clarifier(q: str) -> str:
        seen["q"] = q
        return "能跑通就算完"

    zt = Zhongtang(llm, max_attempts=1, on_event=events.append)
    order = zt.issue_order("做个朝堂页面", clarifier=clarifier)

    assert seen["q"] == "验收标准是什么？", seen
    assert llm.calls == 2, f"澄清 1 次 + 拆单 1 次，实际 {llm.calls}"
    assert "能跑通就算完" in llm.prompts[1], "主公的回答没有进拆单提示词"
    assert order.goal == "做朝堂场景"
    assert any(e["type"] == "order/clarified" for e in events)


def test_no_question_skips_clarifier():
    """模型说「无疑问」→ 不该去打扰主公。"""
    llm = ScriptedLLM(["无疑问", GOOD])
    asked: list[str] = []
    zt = Zhongtang(llm, max_attempts=1)

    zt.issue_order("做个朝堂页面", clarifier=lambda q: asked.append(q) or "")

    assert asked == [], f"不该问却问了：{asked}"
    assert llm.calls == 2


def test_clarify_failure_does_not_block():
    """澄清那一步炸了（模型/网络）→ 按原需求继续拆，不能把流程卡死。"""

    class ClarifyBoom:
        def __init__(self) -> None:
            self.calls = 0

        def quick(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("澄清这一步炸了")
            return GOOD

        def chat(self, prompt: str) -> str:  # pragma: no cover - 不该走到
            raise AssertionError("快路可用时不应回退到 chat")

    llm = ClarifyBoom()
    order = Zhongtang(llm, max_attempts=1).issue_order("做个朝堂页面", clarifier=lambda q: "不影响")

    assert order.goal == "做朝堂场景"
    assert llm.calls == 2


def test_empty_answer_keeps_requirement():
    """主公跳过（空回答）→ 需求原样送进拆单，不拼一个空的「主公补充」。"""
    llm = ScriptedLLM(["要几个司？", GOOD])
    Zhongtang(llm, max_attempts=1).issue_order("做个朝堂页面", clarifier=lambda q: "   ")

    assert "主公补充" not in llm.prompts[1]


# ---------------------------------------------------------------- 跑起来


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  ✗ {name}")
            traceback.print_exc()
            failed += 1

    print()
    print(f"通过 {passed} / {len(tests)}，失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
