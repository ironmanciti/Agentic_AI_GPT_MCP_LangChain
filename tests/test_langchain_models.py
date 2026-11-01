"""03/04장 LangChain·Deep Agent 모델 설정 검증.

추론 모델(gpt-6-luna 등)은 /v1/chat/completions에서 도구 호출을 거부하므로
모든 모델을 use_responses_api=True + reasoning low로 만들고,
응답 텍스트는 .content(블록 리스트) 대신 .text로 읽는지 검사한다.

실제 API 호출 테스트는 비용이 들므로 RUN_LIVE_API=1 일 때만 실행한다.
"""

import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("03_LangChain_Agent", "04_Deep_Agent")
SKIP_DIRS = {".ipynb_checkpoints", "__pycache__"}
IMPORT_LINE = "from langchain.chat_models import init_chat_model"
REQUIRED_OPTIONS = ("use_responses_api=True", 'reasoning={"effort": "low"}')
# .content 읽기 허용 목록 — AI 메시지가 아닌 객체 (Deep Agent 파일 다운로드 응답)
CONTENT_ALLOWED = ("files[0].content",)


def _source_files():
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*")):
            if SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
                continue
            if path.suffix in (".py", ".ipynb"):
                yield path


def _code_cells(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [text]
    nb = json.loads(text)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find_calls(src: str, name: str) -> list[str]:
    """src에서 name( ... ) 호출 전체 문자열을 괄호 짝을 맞춰 추출"""
    calls = []
    for m in re.finditer(rf"\b{name}\(", src):
        depth, i = 0, m.end() - 1
        while i < len(src):
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            if depth == 0:
                calls.append(src[m.start(): i + 1])
                break
            i += 1
    return calls


def missing_options(call: str) -> list[str]:
    return [opt for opt in REQUIRED_OPTIONS if opt not in call]


# .content 직접 읽기 또는 getattr(x, "content", None) or ... 우회 읽기
CONTENT_READ_RE = re.compile(r"\.content\b|getattr\(\w+, [\"']content[\"'], None\)")


def is_content_read(line: str) -> bool:
    return bool(CONTENT_READ_RE.search(line)) and not any(a in line for a in CONTENT_ALLOWED)


def _all_calls():
    return [
        (path, call)
        for path in _source_files()
        for cell in _code_cells(path)
        for call in find_calls(cell, "init_chat_model")
    ]


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------
def test_every_chat_model_uses_responses_api_and_low_effort():
    calls = _all_calls()
    assert len(calls) >= 26, f"init_chat_model 호출 수가 너무 적음: {len(calls)}"
    offenders = [f"{p.relative_to(ROOT)}: {missing_options(c)}" for p, c in calls if missing_options(c)]
    assert offenders == [], "옵션 누락:\n" + "\n".join(offenders)


def test_deep_agents_receive_model_object_not_string():
    offenders = [
        str(p.relative_to(ROOT))
        for p in _source_files()
        for cell in _code_cells(p)
        for call in find_calls(cell, "create_deep_agent")
        if re.search(r"model\s*=\s*f?[\"']openai:", call)
    ]
    assert offenders == [], "문자열 모델(추론 강도 미적용):\n" + "\n".join(offenders)


def test_init_chat_model_imported_before_use():
    for path in _source_files():
        imported = False
        for cell in _code_cells(path):
            if "init_chat_model(" in cell and not imported:
                head = cell[: cell.index("init_chat_model(")]
                assert IMPORT_LINE in head, f"{path.relative_to(ROOT)}: import 전에 init_chat_model 사용"
            imported |= IMPORT_LINE in cell


def test_no_message_content_reads():
    offenders = []
    for path in _source_files():
        for cell in _code_cells(path):
            for line in cell.splitlines():
                if is_content_read(line):
                    offenders.append(f"{path.relative_to(ROOT)}: {line.strip()}")
    assert offenders == [], ".content 대신 .text 사용:\n" + "\n".join(offenders)


def test_final_answer_printed_as_text():
    # 최종 답변은 print(...text)로 출력 (pretty_print는 블록 리스트를 그대로 보여줌)
    pat = re.compile(r"""\b(result|resumed)\[["']messages["']\]\[-1\]\.pretty_print\(\)""")
    offenders = [
        str(p.relative_to(ROOT)) for p in _source_files() for cell in _code_cells(p) if pat.search(cell)
    ]
    assert offenders == [], "최종 답변 pretty_print:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# 경계값: 여러 줄 호출, 중첩 괄호, 인자 사이 주석
# ---------------------------------------------------------------------------
def test_find_calls_handles_multiline_and_nested_parens():
    src = (
        "m = init_chat_model(\n"
        '    f"openai:{os.environ[\'X\']}",   # 주석 (괄호)\n'
        "    use_responses_api=True,\n"
        '    reasoning={"effort": "low"},\n'
        ")\n"
        "other(init_chat_model(a, b=dict(c=1)))\n"
    )
    calls = find_calls(src, "init_chat_model")
    assert len(calls) == 2
    assert missing_options(calls[0]) == []
    assert calls[1] == "init_chat_model(a, b=dict(c=1))"


# ---------------------------------------------------------------------------
# 실패 케이스: 검출기가 잘못된 코드를 잡아내는지
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("src", "missing"),
    [
        ('init_chat_model(os.environ["OPENAI_MODEL"], model_provider="openai")', list(REQUIRED_OPTIONS)),
        ("init_chat_model(m, use_responses_api=True)", ['reasoning={"effort": "low"}']),
        ('init_chat_model(m, use_responses_api=False, reasoning={"effort": "low"})', ["use_responses_api=True"]),
    ],
)
def test_detector_flags_missing_options(src, missing):
    assert missing_options(find_calls(src, "init_chat_model")[0]) == missing


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ('print(result["messages"][-1].content)', True),
        ('text = getattr(last, "content", None) or getattr(last, "text", str(last))', True),
        ("return response.content", True),
        ('print(result["messages"][-1].text)', False),
        ('HumanMessage(content="질문")', False),
        ('{"role": "user", "content": q}', False),
        ("len(files[0].content)", False),
    ],
)
def test_content_read_detector(line, flagged):
    assert is_content_read(line) is flagged


# ---------------------------------------------------------------------------
# 실제 API 호출 (RUN_LIVE_API=1 일 때만)
# ---------------------------------------------------------------------------
live = pytest.mark.skipif(os.environ.get("RUN_LIVE_API") != "1", reason="RUN_LIVE_API=1 일 때만 실행")


@pytest.fixture
def live_model():
    pytest.importorskip("langchain_openai")
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    load_dotenv(ROOT / ".env")
    return init_chat_model(
        os.environ["OPENAI_MODEL"],
        model_provider="openai",
        use_responses_api=True,
        reasoning={"effort": "low"},
    )


@live
def test_live_agent_calls_tool(live_model):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """도시의 현재 기온을 반환한다."""
        return f"{city}: 23.7도"

    agent = create_agent(model=live_model, tools=[get_weather])
    out = agent.invoke({"messages": [{"role": "user", "content": "부산 기온 알려줘"}]})
    assert [m.name for m in out["messages"] if m.type == "tool"] == ["get_weather"]
    assert "23.7" in out["messages"][-1].text


@live
def test_live_plain_invoke_text_is_str(live_model):
    reply = live_model.invoke("한 단어로 인사해줘")
    assert isinstance(reply.content, list)  # Responses API: 블록 리스트
    assert isinstance(reply.text, str) and reply.text.strip()


@live
def test_live_deep_agent_runs(live_model):
    pytest.importorskip("deepagents")
    from deepagents import create_deep_agent

    agent = create_deep_agent(model=live_model, system_prompt="짧게 답한다.")
    out = agent.invoke({"messages": [{"role": "user", "content": "1+1은?"}]})
    assert "2" in out["messages"][-1].text
