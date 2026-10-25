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

from orchestrator.llm import FastPathUnavailable, LLMConfig  # noqa: E402
from orchestrator.orchestrator import OrderRejected, Zhongtang, parse_question  # noqa: E402
from orchestrator.attest import parse_stance, run_attest  # noqa: E402
from orchestrator.verify import audit, parse_verdict, run_verify  # noqa: E402
from orchestrator.summary import parse_summary, run_summary  # noqa: E402
from orchestrator.schema import ParseError, parse_order, validate  # noqa: E402
from orchestrator.execute import (  # noqa: E402
    allocate_case_id,
    choose_filename,
    execute_order,
    strip_model_text,
)
from stage.cloud import (  # noqa: E402
    PLAN_BASE,
    CloudError,
    CloudLink,
    config_for,
    menu_view,
    model_ids_from,
    prepare_link,
    remember,
    load_links,
)
from stage.server import unknown_model_message  # noqa: E402


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

    def quick(self, prompt: str, **kwargs) -> str:
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


def test_unknown_model_is_rejected_when_list_is_known():
    """端点名单取到了，就不许用名单以外的名字。"""
    msg = unknown_model_message("not-a-real-model", ["qwen3.8-27b"])
    assert msg is not None
    assert "qwen3.8-27b" in msg
    assert unknown_model_message("qwen3.8-27b", ["qwen3.8-27b"]) is None
    assert unknown_model_message("", ["qwen3.8-27b"]) is None


def test_unknown_model_is_allowed_when_list_failed():
    """名单没取到时不拦，避免端点临时失败把升堂卡死。"""
    assert unknown_model_message("qwen3.8-27b", None) is None


def test_empty_answer_keeps_requirement():
    """主公跳过（空回答）→ 需求原样送进拆单，不拼一个空的「主公补充」。"""
    llm = ScriptedLLM(["要几个司？", GOOD])
    Zhongtang(llm, max_attempts=1).issue_order("做个朝堂页面", clarifier=lambda q: "   ")

    assert "主公补充" not in llm.prompts[1]


def test_case_folder_gets_a_new_number():
    """同一天再升一堂，不覆盖上一场的案卷。"""
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    (root / "20261024-01").mkdir()
    assert allocate_case_id(root, "20261024") == "20261024-02"


def test_execute_writes_files_and_product_page():
    """五司的正文落进各司夹子，成品页能打开，刑部只凭文件在不在下判。"""
    import tempfile
    from pathlib import Path

    from orchestrator.schema import Order, Task

    class Pen:
        def quick(self, prompt, **kwargs):
            if "做升日动画" in prompt:
                return "<!DOCTYPE html><html><body><h1>升日</h1></body></html>"
            return "# 说明\n\n这一条已经写成文件，可以打开核对。"

    order = Order(
        goal="做一页能打开的升日动画",
        tasks=[
            Task("礼部", "写一句画面说明", "画面说明"),
            Task("兵部", "做升日动画", "升日动画"),
            Task("工部", "写一句如何打开", "打开说明"),
        ],
    )
    root = Path(tempfile.mkdtemp())
    events = []
    result = execute_order(
        Pen(), order, "做个动画", root, on_event=events.append, order_no="20261024-01"
    )

    assert result.ok
    assert result.product.is_file()
    assert result.product.parent == root
    assert result.product.suffix == ".html"
    names = sorted(p.name for p in root.iterdir())
    assert names == [result.product.name]
    assert not (root / "升堂单.md").exists()
    assert events[-1]["type"] == "execute/final"


def test_empty_delivery_is_rejected_without_a_file():
    """交回来是空的，刑部退回，而且不能冒出一份假文件。"""
    import tempfile
    from pathlib import Path

    from orchestrator.schema import Order, Task

    class Empty:
        def quick(self, prompt, **kwargs):
            return "好的"

    order = Order(
        goal="写一份说明",
        tasks=[
            Task("礼部", "写说明", "说明"),
            Task("兵部", "写做法", "做法"),
            Task("工部", "写如何打开", "打开"),
        ],
    )
    root = Path(tempfile.mkdtemp())
    result = execute_order(Empty(), order, "写说明", root, order_no="20261024-03")
    assert not result.ok
    assert all(not item.ok for item in result.checks)
    assert list(root.iterdir()) == []


def test_strip_fence_and_choose_html_name():
    body = strip_model_text("说明如下：\n```html\n<!DOCTYPE html><p>戏</p>\n```\n以上。")
    assert body.startswith("<!DOCTYPE html>")
    name = choose_filename("升日动画", body, set())
    assert name.endswith(".html")


def test_plan_link_uses_token_plan_address():
    """Token Plan 不听调用方传的地址，免得钥匙打到别的站。"""
    kind, _label, base = prepare_link("plan", "sk-sp-abc", "https://evil.example/v1")
    assert kind == "plan"
    assert base == PLAN_BASE


def test_plan_rejects_pay_as_you_go_key():
    """按量的 sk- 不能塞进订阅这一路。"""
    try:
        prepare_link("plan", "sk-normal", None)
    except CloudError as exc:
        assert "sk-sp-" in str(exc)
        return
    raise AssertionError("应当拒绝")


def test_api_defaults_to_deepseek_and_strips_suffix():
    """地址空着走 DeepSeek；把聊天路径贴进来时剥掉末尾。"""
    _kind, _label, base = prepare_link("api", "sk-xxx", "")
    assert base == "https://api.deepseek.com/v1"
    _kind, _label, base = prepare_link("api", "sk-xxx", "https://example.com/v1/chat/completions")
    assert base == "https://example.com/v1"


def test_api_rejects_non_http():
    try:
        prepare_link("api", "sk-xxx", "file:///etc/passwd")
    except CloudError:
        return
    raise AssertionError("应当拒绝")


def test_saved_key_stays_out_of_the_menu():
    """钥匙落在本机文件里，给页面的名单里没有它。"""
    import json
    import tempfile

    store = Path(tempfile.mkdtemp()) / "cloud.json"
    remember(
        CloudLink("api", "云端接口", "https://api.deepseek.com/v1", "sk-secret", ["deepseek-chat"]),
        store,
    )
    menu = menu_view(load_links(store))
    assert "sk-secret" not in json.dumps(menu)
    assert menu[0]["models"] == ["deepseek-chat"]
    assert load_links(store)["api"].api_key == "sk-secret"


def test_cloud_config_points_at_that_key():
    """选了云端模型，这一单改地址和钥匙，本机那份配置不动。"""
    link = CloudLink("plan", "Token Plan", PLAN_BASE, "sk-sp-1", ["qwen3.6-plus"])
    cfg = LLMConfig(
        model="local-model",
        base_url="http://127.0.0.1:8096/v1",
        api_key="local",
        reasoning="off",
    )
    got = config_for(cfg, link, "qwen3.6-plus")
    assert got.model == "qwen3.6-plus"
    assert got.fast_base_url == PLAN_BASE
    assert got.fast_api_key == "sk-sp-1"
    assert got.reasoning is None
    assert cfg.api_key == "local"
    assert cfg.model == "local-model"


def test_parse_openai_model_list():
    ids = model_ids_from({"data": [{"id": "a"}, {"id": ""}, {"id": "a"}, "nope", {"name": "b"}]})
    assert ids == ["a"]


# ---------------------------------------------------------------- 附议


AGREE_JSON = '{"表态":"附议"}'


def _order_three():
    from orchestrator.schema import Order, Task

    return Order(
        goal="做一页能打开的升日动画",
        tasks=[
            Task("礼部", "写一句画面说明", "画面说明"),
            Task("兵部", "做升日动画", "升日动画"),
            Task("工部", "写一句如何打开", "打开说明"),
        ],
    )


def test_attest_all_agree_passes_in_one_round():
    """五司全附议：一轮收口，单子一个字不动。刑部不参加。"""
    llm = ScriptedLLM([AGREE_JSON] * 5)
    events = []
    result = run_attest(llm, _order_three(), on_event=events.append)

    assert result.passed
    assert result.rounds == 1
    assert llm.fast_calls == 5
    assert result.amendments == []
    assert events[-1]["type"] == "attest/final"
    assert events[-1]["passed"] is True


def test_attest_dissent_amends_order_and_second_round_passes():
    """兵部提异议 → 真改单 → 第二轮全附议。改的是单子本身，不只是屏幕。"""
    dissent = '{"表态":"异议","理由":"工期不够","改第几条":2,"改成什么":"先出静态场景"}'
    replies = [AGREE_JSON, AGREE_JSON, AGREE_JSON, dissent, AGREE_JSON] + [AGREE_JSON] * 5
    llm = ScriptedLLM(replies)
    result = run_attest(llm, _order_three())

    assert result.passed
    assert result.rounds == 2
    assert len(result.amendments) == 1
    assert result.order.tasks[1].action == "先出静态场景"
    # 没给新交付物就保留原样，不许把它改成空
    assert result.order.tasks[1].deliverable == "升日动画"
    # 别的条一个字都不许动
    assert result.order.tasks[0].action == "写一句画面说明"
    assert result.order.tasks[2].action == "写一句如何打开"


def test_attest_second_round_sees_the_amendment():
    """第二轮提示词里要写明上一轮改了什么，否则它会重复提同一个异议。"""
    dissent = '{"表态":"异议","理由":"工期不够","改第几条":2,"改成什么":"先出静态场景"}'
    replies = [AGREE_JSON, AGREE_JSON, AGREE_JSON, dissent, AGREE_JSON] + [AGREE_JSON] * 5
    llm = ScriptedLLM(replies)
    run_attest(llm, _order_three())

    second = llm.prompts[5]
    assert "已按上一轮异议改过" in second
    assert "先出静态场景" in second


def test_attest_vague_dissent_falls_back_to_agree():
    """提了异议却没说清改哪条 → 不算数，降级为附议，不拖流程。"""
    vague = '{"表态":"异议","理由":"感觉不太行"}'
    llm = ScriptedLLM([AGREE_JSON, AGREE_JSON, AGREE_JSON, vague, AGREE_JSON])
    result = run_attest(llm, _order_three())

    assert result.passed
    assert result.rounds == 1
    assert result.amendments == []
    bing = [s for s in result.stances if s.dept == "兵部"][0]
    assert bing.verdict == "附议"
    assert "没说清" in bing.reason


def test_attest_two_rounds_without_consensus_escalates():
    """两轮都吵不下来 → 上抛主公，不许无限吵。"""
    first = '{"表态":"异议","理由":"不行","改第几条":2,"改成什么":"做法甲"}'
    second = '{"表态":"异议","理由":"还是不行","改第几条":2,"改成什么":"做法乙"}'
    replies = (
        [AGREE_JSON] * 3 + [first] + [AGREE_JSON]
        + [AGREE_JSON] * 3 + [second] + [AGREE_JSON]
    )
    llm = ScriptedLLM(replies)
    result = run_attest(llm, _order_three())

    assert not result.passed
    assert result.escalate
    assert result.rounds == 2
    assert llm.fast_calls == 10


def test_attest_model_failure_does_not_block():
    """某个司的模型调用炸了 → 跳过它，不当异议，整场照常收口。"""

    class Flaky:
        def __init__(self):
            self.n = 0

        def quick(self, prompt, **kwargs):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("端点没应")
            return AGREE_JSON

    events = []
    result = run_attest(Flaky(), _order_three(), on_event=events.append)

    assert result.passed
    assert any(e["type"] == "attest/skip" for e in events)


def test_attest_bad_amendment_is_not_applied():
    """改完反而不合格（交付物写空话）→ 这一条不改，也不许把坏单子往下走。"""
    bad = '{"表态":"异议","理由":"要改","改第几条":2,"改成什么":"重做","交付物改成":"研究一下"}'
    llm = ScriptedLLM([AGREE_JSON, AGREE_JSON, AGREE_JSON, bad, AGREE_JSON])
    result = run_attest(llm, _order_three())

    assert not result.passed
    assert result.amendments == []
    assert result.order.tasks[1].deliverable == "升日动画"


def test_parse_stance_handles_fenced_and_numbered():
    """模型爱包代码块、爱写「第 3 条」——解析层要忍。"""
    raw = '```json\n{"表态":"异议","改第几条":"第 3 条","改成什么":"先做静态页"}\n```'
    stance = parse_stance(raw, "兵部")

    assert stance.verdict == "异议"
    assert stance.target == 3
    assert stance.valid


def test_parse_stance_no_dissent_is_agree():
    """「无异议」里含「异议」两个字，判序不能反。"""
    stance = parse_stance('{"表态":"无异议"}', "户部")
    assert stance.verdict == "附议"


def test_attest_prompt_is_per_department():
    """六司共用一句话，真调模型也只会齐声附议。视角必须按司分开。"""
    from orchestrator.prompts import attest_prompt

    order = _order_three()
    hubu = attest_prompt(order, "户部")
    bingbu = attest_prompt(order, "兵部")

    assert "资源够不够" in hubu
    assert "技术上做不做得出来" in bingbu
    assert "刑部" not in hubu


# ---------------------------------------------------------------- 勘验


def _order_one():
    from orchestrator.schema import Order, Task

    return Order(goal="做一页升日动画", tasks=[Task("兵部", "做升日动画", "升日动画")])


def _execution_one(body="<!DOCTYPE html><html><body>日出</body></html>"):
    from pathlib import Path

    from orchestrator.execute import Check, Execution

    return Execution(
        "20261025-01",
        Path("."),
        Path("."),
        [Check(0, "兵部", "升日动画", "", True, "待收成一件成品", body)],
    )


def test_parse_verdict_handles_no_go():
    """「不过」里含「过」，判序不能反。"""
    assert parse_verdict('{"核":"过"}') == (True, "")
    ok, reason = parse_verdict('{"核":"退","理由":"没有证据"}')
    assert ok is False
    assert "证据" in reason
    assert parse_verdict('{"核":"不过"}')[0] is False
    assert parse_verdict("看不懂") is None


def test_verify_skips_model_when_objective_fails():
    """执行阶段就没交出来 → 客观层直接退，不浪费一次模型调用。"""
    from pathlib import Path

    from orchestrator.execute import Check, Execution

    class Silent:
        def __init__(self):
            self.calls = 0

        def quick(self, prompt, **kwargs):
            self.calls += 1
            return '{"核":"过"}'

    llm = Silent()
    execution = Execution(
        "x", Path("."), Path("."), [Check(0, "兵部", "升日动画", "", False, "交回来的是空的", "")]
    )
    result = run_verify(llm, _order_one(), execution, allow_retry=False)

    assert not result.passed
    assert llm.calls == 0
    assert result.verdicts[0].source == "客观"


def test_verify_reads_content_not_just_files():
    """客观层过了还要看内容：正文真送进提示词，刑部说退就得退。"""

    class Judge:
        def quick(self, prompt, **kwargs):
            assert "日出" in prompt
            assert "程序已经查过" in prompt
            return '{"核":"退","理由":"只有一句话，没有可复核的做法"}'

    result = run_verify(Judge(), _order_one(), _execution_one(), allow_retry=False)

    assert not result.passed
    assert "可复核" in result.verdicts[0].reason
    assert result.verdicts[0].source == "客观+刑部"


def test_verify_retries_once_and_records_it():
    """退回 → 真补交 → 再核 → 过。补交这件事要记在结果里。"""

    class Retry:
        def __init__(self):
            self.n = 0

        def quick(self, prompt, **kwargs):
            self.n += 1
            if "补上这一点" in prompt:
                return (
                    "<!DOCTYPE html><html><body>"
                    "日出：先画地平线，再逐帧抬升太阳，共 24 帧。"
                    "</body></html>"
                )
            if "程序已经查过" in prompt:
                if "24 帧" in prompt:
                    return '{"核":"过"}'
                return '{"核":"退","理由":"太短，没有做法"}'
            raise AssertionError("意料之外的调用")

    llm = Retry()
    result = run_verify(llm, _order_one(), _execution_one(), allow_retry=True)

    assert result.passed
    assert result.verdicts[0].retried
    assert llm.n == 3  # 核 → 补交 → 再核


def test_verify_falls_back_to_facts_when_judge_silent():
    """刑部没应 → 按客观放行，但如实标「刑部未应」，不许假装核过。"""

    class Dead:
        def quick(self, prompt, **kwargs):
            raise RuntimeError("端点没应")

    events = []
    result = run_verify(
        Dead(), _order_one(), _execution_one(), on_event=events.append, allow_retry=False
    )

    assert result.passed
    assert "刑部未应" in result.verdicts[0].source
    assert any(e["type"] == "verify/skip" for e in events)


def test_audit_reports_html_completeness():
    """客观层要说出「是不是完整页面」，这是刑部下判的依据。"""
    import tempfile
    from pathlib import Path

    from orchestrator.execute import Check

    path = Path(tempfile.mkdtemp()) / "page.html"
    path.write_text("<!DOCTYPE html><html><body>x</body></html>", "utf-8")
    check = Check(0, "兵部", "页面", str(path), True, "成品已落盘", "<html>")

    facts = audit(check)
    assert any("完整页面" in item for item in facts)


def test_close_partial_html_finishes_truncated_page():
    """模型被截断时补个收尾，至少能打开；完整的页面一个字都不许动。"""
    from orchestrator.execute import close_partial_html

    whole = "<!DOCTYPE html><html><body>x</body></html>"
    assert close_partial_html(whole) == (whole, False)

    half = "<!DOCTYPE html><html><body><h1>日出"
    body, patched = close_partial_html(half)
    assert patched
    assert body.rstrip().endswith("</html>")
    assert "<h1>日出" in body

    # 根本不是 HTML，不许硬补
    assert close_partial_html("一段说明") == ("一段说明", False)


def test_verify_prompt_marks_excerpt_as_excerpt():
    """给刑部看的是节选，必须说明白——否则它把节选当成文件被截断，好活也判退。"""
    from orchestrator.prompts import verify_prompt

    full = verify_prompt(
        "目标", "兵部", "做页面", "页面", ["写到了 </html>，是完整页面"], "x" * 100, cut=True
    )
    assert "节选，不是文件被截断" in full
    assert "完整页面" in full

    plain = verify_prompt("目标", "兵部", "做页面", "页面", [], "x" * 100, cut=False)
    assert "节选，不是文件被截断" not in plain


# ---------------------------------------------------------------- 汇总


def test_parse_summary_reads_all_four_parts():
    """结案要四样齐全：结语、结论、风险、未做。缺了「未做」就是只报喜。"""
    raw = '{"结语":"做成了","结论":["第1条过"],"风险":["动画只有24帧"],"未做":["没有音效"]}'
    summary = parse_summary(raw)

    assert summary.verdict == "做成了"
    assert summary.points == ["第1条过"]
    assert summary.risks == ["动画只有24帧"]
    assert summary.undone == ["没有音效"]
    assert summary.source == "中堂"


def test_parse_summary_rejects_empty():
    assert parse_summary("看不懂") is None
    assert parse_summary('{"别的":"东西"}') is None


def test_summary_falls_back_when_model_silent():
    """中堂没写出结案 → 程序版接手，但整场必须有结论。"""
    from pathlib import Path

    from orchestrator.execute import Check, Execution
    from orchestrator.verify import Inspection, Verdict

    class Dead:
        def quick(self, prompt, **kwargs):
            raise RuntimeError("端点没应")

    execution = Execution(
        "20261025-01",
        Path("."),
        Path("."),
        [Check(0, "兵部", "升日动画", "", True, "待收成一件成品", "<html></html>")],
    )
    inspection = Inspection(
        False, [Verdict(0, "兵部", "升日动画", False, "没证据", [], "客观+刑部")]
    )
    events = []
    summary = run_summary(
        Dead(), _order_one(), "做一页升日动画", execution, inspection, on_event=events.append
    )

    assert summary.source == "程序"
    assert summary.undone
    assert any(e["type"] == "summary/fallback" for e in events)
    assert events[-1]["type"] == "summary/final"


def test_summary_writes_case_file_outside_product_dir():
    """结案文书要落盘，但绝不进成品夹。"""
    import tempfile
    from pathlib import Path

    from orchestrator.execute import Check, Execution
    from orchestrator.verify import Inspection, Verdict

    class Zhongtang:
        def quick(self, prompt, **kwargs):
            assert "刑部勘验" in prompt
            return '{"结语":"做成了","结论":["第1条过"],"风险":[],"未做":[]}'

    execution = Execution(
        "20261025-01",
        Path("."),
        Path("."),
        [Check(0, "兵部", "升日动画", "", True, "成品已落盘", "<html></html>")],
    )
    inspection = Inspection(
        True, [Verdict(0, "兵部", "升日动画", True, "", [], "客观+刑部")]
    )
    out = Path(tempfile.mkdtemp()) / "结案"
    summary = run_summary(
        Zhongtang(), _order_one(), "做一页升日动画", execution, inspection, out_dir=out
    )

    assert summary.path
    text = Path(summary.path).read_text("utf-8")
    assert "结案" in text
    assert "做成了" in text
    assert "## 风险" in text
    assert "## 未做" in text


def test_product_prefers_visual_over_longest():
    """成品挑「最像成品」的那一件，不是最长的那一件。

    实测踩过：吏部写了个 7868 字符的「升堂单」HTML，把兵部 3291 字符的动画页挤掉，
    结案说「升日动画已交付」，成品却是任务清单。
    """
    import tempfile
    from pathlib import Path

    from orchestrator.schema import Order, Task

    long_list = "<!DOCTYPE html><html><body>" + "清单条目。" * 2000 + "</body></html>"
    short_anim = "<!DOCTYPE html><html><body><h1>日出</h1></body></html>"

    class Pen:
        def quick(self, prompt, **kwargs):
            if "升堂单" in prompt:
                return long_list
            return short_anim

    order = Order(
        goal="做一页升日动画",
        tasks=[
            Task("吏部", "写一张升堂单", "升堂单"),
            Task("兵部", "做一页升日动画", "升日动画"),
        ],
    )
    root = Path(tempfile.mkdtemp())
    result = execute_order(Pen(), order, "做一页升日动画", root, order_no="20261025-02")

    assert result.product.is_file()
    assert "升日动画" in result.product.name
    assert len(result.product.read_bytes()) < len(long_list.encode("utf-8"))


def test_product_skips_paper_deliverables():
    """交付物是「清单/说明」的，就算写了 HTML、就算最长，也不许当成品。

    实测踩过：礼部的活是「写一页升日动画的文案清单」（含「动画」但也含「清单」），
    它交了一整个动画页，因为最长被选中，结案就说成了「动画已交付」。
    """
    import tempfile
    from pathlib import Path

    from orchestrator.schema import Order, Task

    long_paper = "<!DOCTYPE html><html><body>" + "文案条目。" * 2000 + "</body></html>"
    short_anim = "<!DOCTYPE html><html><body><h1>日出</h1></body></html>"

    class Pen:
        def quick(self, prompt, **kwargs):
            if "文案" in prompt:
                return long_paper
            return short_anim

    order = Order(
        goal="做一页升日动画",
        tasks=[
            Task("礼部", "写一页升日动画的文案清单", "文案清单（含标题、按钮文字）"),
            Task("兵部", "做一页升日动画", "升日动画"),
        ],
    )
    root = Path(tempfile.mkdtemp())
    result = execute_order(Pen(), order, "做一页升日动画", root, order_no="20261025-03")

    assert result.product.is_file()
    assert "升日动画" in result.product.name
    assert len(result.product.read_bytes()) < len(long_paper.encode("utf-8"))
    # 没当成品的那条不许记成品路径，否则刑部会以为每条都落了这个文件
    merged = [c for c in result.checks if c.note == "已并入这一件成品"]
    assert merged and all(not c.path for c in merged)


def test_task_prompt_asks_for_docs_inside_the_page():
    """交付物要「HTML + 使用说明」时，得让它把说明写进页面。

    原来提示词一律写着「不写说明文字」，这种活就永远交不出来（实测踩过）。
    """
    from orchestrator.execute import task_prompt
    from orchestrator.schema import Order

    order = Order(goal="做动画", tasks=[])
    prompt = task_prompt(order, "工部", "打包成 HTML 并附使用说明", "1个HTML文件及1份使用说明")
    assert "页面末尾加一节" in prompt
    assert "不写注释、不写多余说明" not in prompt

    plain = task_prompt(order, "兵部", "做一页动画", "升日动画")
    assert "不写注释、不写多余说明" in plain


def test_html_body_is_not_thrown_away_as_side_paper():
    """交付物写着「附使用说明」，但正文是一整个页面——不能当纸丢掉。"""
    import tempfile
    from pathlib import Path

    from orchestrator.schema import Order, Task

    page = (
        "<!DOCTYPE html><html><body><h1>日出</h1>"
        "<section>使用说明：点开始按钮</section></body></html>"
    )

    class Pen:
        def quick(self, prompt, **kwargs):
            return page

    order = Order(
        goal="做动画", tasks=[Task("工部", "打包成HTML并附使用说明", "1个HTML文件及1份使用说明")]
    )
    root = Path(tempfile.mkdtemp())
    result = execute_order(Pen(), order, "做动画", root, order_no="20261025-04")

    assert result.product.is_file()
    assert "使用说明" in result.product.read_text("utf-8")


def test_summary_tells_apart_never_delivered_and_retry_failed():
    """「没交」和「补交过但没过」是两回事。混了，结案就会写成「未附交」。"""
    from orchestrator.prompts import summary_prompt
    from orchestrator.verify import Inspection, Verdict

    inspection = Inspection(
        False,
        [Verdict(0, "工部", "说明", False, "缺使用说明", [], "客观+刑部", retried=True)],
    )
    text = summary_prompt("做动画", _order_one(), None, inspection)

    assert "补交过，仍不合格" in text


def test_summary_adds_retry_note_when_model_forgets():
    """模型把「补交过但没过」写成「未交」时，程序补一句——事实不能被抹掉。"""
    from orchestrator.summary import Summary, _ensure_retry_note
    from orchestrator.verify import Inspection, Verdict

    inspection = Inspection(
        False, [Verdict(0, "礼部", "清单", False, "缺清单", [], "客观+刑部", retried=True)]
    )
    summary = Summary(
        verdict="已交付", points=[], risks=[], undone=["未交付清单文件"], source="中堂"
    )
    out = _ensure_retry_note(summary, inspection)

    assert any("补交过一轮" in item for item in out.undone)


def test_summary_does_not_duplicate_retry_note():
    """模型已经写清楚补交过，就别再补一遍。"""
    from orchestrator.summary import Summary, _ensure_retry_note
    from orchestrator.verify import Inspection, Verdict

    inspection = Inspection(
        False, [Verdict(0, "礼部", "清单", False, "缺清单", [], "客观+刑部", retried=True)]
    )
    summary = Summary(verdict="已交付", undone=["第 1 条补交过仍不合格"], source="中堂")
    out = _ensure_retry_note(summary, inspection)

    assert len(out.undone) == 1


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
