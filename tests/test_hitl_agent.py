"""104_HITL_Agent 노트북 검증: SQL 도구 + Human-in-the-Loop 흐름 (approve / edit / reject / allowed_decisions).

노트북의 실제 셀 코드를 실행하되, 모델은 정해진 도구 호출을 차례로 내놓는 가짜 모델로 바꿔
API 호출 없이 HITL 동작을 확인한다.
"""

import json
import sqlite3
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "03_LangChain_Agent"
MAIN = FOLDER / "104_HITL_Agent.ipynb"
ANSWER = FOLDER / "practice_answer_104_HITL_Agent.ipynb"
AGENT_MARKER = "\nagent = create_agent("  # 본문 HITL 에이전트 셀


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_with(marker: str, path: Path = MAIN) -> str:
    found = [c for c in _code_cells(path) if marker in c]
    assert len(found) == 1, f"{path.name}: '{marker}' 셀 {len(found)}개"
    return found[0]


class ScriptedModel(GenericFakeChatModel):
    """정해진 AIMessage 를 차례로 반환하는 가짜 모델 (도구 바인딩은 무시)"""

    def bind_tools(self, tools, **kwargs):
        return self


def call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "sql_db_query", "args": {"query": query}, "id": call_id}])


def answer(text: str) -> AIMessage:
    return AIMessage(content=text)


def make_ns(monkeypatch, *script: AIMessage) -> dict:
    """본문 노트북의 도구·에이전트·도우미 셀을 가짜 모델로 실행한 네임스페이스"""
    monkeypatch.chdir(FOLDER)
    # Command 는 노트북 4.1 셀에서 import 되므로 미리 넣어 둠
    ns = {"model": ScriptedModel(messages=iter(script)), "Command": Command}
    for marker in ("def connect_readonly", AGENT_MARKER, "def run("):
        # 저장소의 노트북 셀 코드를 그대로 실행해 검증 (외부 입력이 아니므로 exec 허용)
        exec(_cell_with(marker), ns)  # noqa: S102
    return ns


def ask(ns, agent_name: str, question: str, thread: str):
    config = {"configurable": {"thread_id": thread}}
    return ns["run"](ns[agent_name], {"messages": [{"role": "user", "content": question}]}, config), config


def tool_messages(result) -> list[str]:
    return [m.text for m in result["messages"] if m.type == "tool"]


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------
def test_tools_minimal_set(monkeypatch):
    ns = make_ns(monkeypatch)
    assert [t.name for t in ns["tools"]] == ["sql_db_list_tables", "sql_db_schema", "sql_db_query"]
    assert "Artist" in ns["sql_db_list_tables"].invoke({})
    assert ns["sql_db_schema"].invoke({"table_names": "Artist"}).startswith("CREATE TABLE")
    assert ns["sql_db_query"].invoke({"query": "SELECT COUNT(*) FROM Album"}) == "[(347,)]"


def test_interrupt_before_query_and_approve(monkeypatch, capsys):
    ns = make_ns(monkeypatch, call("SELECT COUNT(*) FROM Album"), answer("앨범은 347개입니다."))
    result, config = ask(ns, "agent", "앨범 수?", "t_approve")
    request = result["__interrupt__"][0].value
    assert request["action_requests"][0]["args"]["query"] == "SELECT COUNT(*) FROM Album"
    assert request["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert tool_messages(result) == []                       # 승인 전에는 실행되지 않음
    assert "⏸ 승인 대기: sql_db_query" in capsys.readouterr().out

    result = ns["run"](ns["agent"], Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert tool_messages(result) == ["[(347,)]"]
    assert "✅ 최종 답변: 앨범은 347개입니다." in capsys.readouterr().out


def test_edit_runs_human_query(monkeypatch):
    ns = make_ns(monkeypatch, call("SELECT COUNT(*) FROM Album"), answer("3개"))
    _, config = ask(ns, "agent", "q", "t_edit")
    edited = "SELECT Name FROM Artist ORDER BY ArtistId LIMIT 2"
    result = ns["run"](ns["agent"], Command(resume={"decisions": [{
        "type": "edit", "edited_action": {"name": "sql_db_query", "args": {"query": edited}},
    }]}), config)
    (message,) = tool_messages(result)
    assert message.endswith("[('AC/DC',), ('Accept',)]")   # 모델 쿼리가 아니라 사람이 고친 쿼리 결과
    assert "human reviewer replaced this tool call" in message  # 모델에게 수정 사실 안내


def test_reject_sends_reason_then_retry(monkeypatch):
    ns = make_ns(
        monkeypatch,
        call("SELECT FirstName, Email FROM Customer", "c1"),
        call("SELECT Country, COUNT(*) FROM Customer GROUP BY Country LIMIT 2", "c2"),
        answer("국가별 고객 수입니다."),
    )
    _, config = ask(ns, "agent", "고객 이메일", "t_reject")
    reason = "개인정보(이메일)는 조회할 수 없습니다."
    result = ns["run"](ns["agent"], Command(resume={"decisions": [{"type": "reject", "message": reason}]}), config)
    assert any(reason in t for t in tool_messages(result))         # 거부 사유가 모델에게 전달됨
    assert "Email" not in result["__interrupt__"][0].value["action_requests"][0]["args"]["query"]

    result = ns["run"](ns["agent"], Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert tool_messages(result)[-1].startswith("[('Argentina', 1)")


# ---------------------------------------------------------------------------
# 경계값: 읽기 도구는 승인 없이 바로 실행
# ---------------------------------------------------------------------------
def test_read_only_tools_run_without_approval(monkeypatch):
    list_call = AIMessage(content="", tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": "l1"}])
    ns = make_ns(monkeypatch, list_call, answer("테이블 목록입니다."))
    result, _ = ask(ns, "agent", "테이블?", "t_list")
    assert "__interrupt__" not in result
    assert "Artist" in tool_messages(result)[0]


# ---------------------------------------------------------------------------
# 실패 케이스
# ---------------------------------------------------------------------------
def test_readonly_connection_blocks_writes_even_if_approved(monkeypatch):
    ns = make_ns(monkeypatch, call("DELETE FROM Artist"), answer("삭제 실패"))
    _, config = ask(ns, "agent", "삭제", "t_write")
    result = ns["run"](ns["agent"], Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert "readonly" in tool_messages(result)[0]
    con = sqlite3.connect(f"file:{FOLDER / 'Chinook.db'}?mode=ro", uri=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM Artist").fetchone() == (275,)
    finally:
        con.close()


def test_answer_notebook_matches_main_setup():
    setup = _cell_with(AGENT_MARKER, ANSWER)
    for marker in ("def connect_readonly", AGENT_MARKER, "def run("):
        body = _cell_with(marker).replace("print(\"도구:\", [t.name for t in tools])", "").strip()
        assert body in setup, f"답안 준비 셀이 본문과 다름: {marker}"
