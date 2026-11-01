"""02_OpenAI_MCP_Agent/mcp_client.py (stdio / HTTP 겸용 클라이언트) 테스트.

에이전트 실행(run_agent)은 OpenAI API를 호출하므로 가짜 함수로 대체하고,
MCP 서버 연결은 실제 stdio 서버를 띄워 도구 목록만 확인한다.
"""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("agents.mcp", reason="openai-agents (MCP 지원 버전) 필요")
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp

ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "02_OpenAI_MCP_Agent"
sys.path.insert(0, str(MCP_DIR))

import mcp_client


@pytest.fixture
def client(monkeypatch):
    """대화 기록을 비운 상태의 Flask 테스트 클라이언트"""
    mcp_client.chat_history.clear()
    monkeypatch.setattr(mcp_client, "TRANSPORT", "stdio")
    yield mcp_client.app.test_client()
    mcp_client.chat_history.clear()


def fake_agent(reply="", tool_calls=(), error=None):
    """run_agent 대체용 가짜 코루틴 함수 (호출 인자를 calls에 기록)"""
    calls = []

    async def _run(messages, transport):
        calls.append((list(messages), transport))
        if error:
            raise error
        return reply, list(tool_calls)

    _run.calls = calls
    return _run


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------
def test_make_server_stdio_uses_mcp_json():
    server = mcp_client.make_server("stdio")
    assert isinstance(server, MCPServerStdio)
    assert server.name == "mcp_server"
    assert server.params.command == sys.executable   # mcp.json의 "python" → 현재 인터프리터
    assert server.params.args == ["mcp_server.py"]
    assert Path(server.params.cwd) == MCP_DIR


def test_make_server_http_uses_server_url():
    server = mcp_client.make_server("http")
    assert isinstance(server, MCPServerStreamableHttp)
    assert server.name == "mcp_server"
    assert server.params["url"] == "http://localhost:8000/mcp"


@pytest.mark.parametrize(
    ("transport", "label", "has_http_hint"),
    [("stdio", "(stdio)", False), ("http", "(Streamable HTTP)", True)],
)
def test_home_shows_transport(client, monkeypatch, transport, label, has_http_hint):
    monkeypatch.setattr(mcp_client, "TRANSPORT", transport)
    res = client.get("/")
    html = res.get_data(as_text=True)
    assert res.status_code == 200
    assert f"<title>의류 &amp; Chinook DB 에이전트 {label}</title>" in html
    assert ("python mcp_server.py --http" in html) is has_http_hint
    assert "{{" not in html and "{%" not in html


@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_chat_returns_reply_and_keeps_history(client, monkeypatch, transport):
    monkeypatch.setattr(mcp_client, "TRANSPORT", transport)
    agent = fake_agent("셔츠는 25000원입니다.", ["get_price"])
    monkeypatch.setattr(mcp_client, "run_agent", agent)

    res = client.post("/chat", json={"message": "셔츠 가격?"})

    assert res.status_code == 200
    assert res.get_json() == {"reply": "셔츠는 25000원입니다.", "tool_calls": ["get_price"]}
    assert agent.calls == [([{"role": "user", "content": "셔츠 가격?"}], transport)]
    assert mcp_client.chat_history[-1] == {"role": "assistant", "content": "셔츠는 25000원입니다."}


def test_chat_sends_full_history(client, monkeypatch):
    agent = fake_agent("답")
    monkeypatch.setattr(mcp_client, "run_agent", agent)
    client.post("/chat", json={"message": "첫 질문"})
    client.post("/chat", json={"message": "두 번째 질문"})
    assert [m["content"] for m in agent.calls[1][0]] == ["첫 질문", "답", "두 번째 질문"]


def test_reset_clears_history(client, monkeypatch):
    monkeypatch.setattr(mcp_client, "run_agent", fake_agent("답"))
    client.post("/chat", json={"message": "질문"})
    res = client.post("/reset")
    assert res.status_code == 200
    assert res.get_json() == {"ok": True}
    assert mcp_client.chat_history == []


def test_stdio_server_exposes_allowed_tools():
    """실제 mcp_server.py를 stdio로 띄워 도구 6개가 노출되는지 확인"""
    pytest.importorskip("fastmcp")

    async def list_tool_names():
        async with mcp_client.make_server("stdio") as server:
            return sorted(t.name for t in await server.list_tools())

    assert asyncio.run(list_tool_names()) == sorted(mcp_client.ALLOWED_TOOLS)


# ---------------------------------------------------------------------------
# 경계값
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "stdio"), (["--http"], "http"), (["--debug", "--http"], "http"), (["http"], "stdio")],
)
def test_parse_transport(argv, expected):
    assert mcp_client.parse_transport(argv) == expected


def test_chat_whitespace_message_rejected(client, monkeypatch):
    agent = fake_agent("답")
    monkeypatch.setattr(mcp_client, "run_agent", agent)
    res = client.post("/chat", json={"message": "   "})
    assert res.status_code == 400
    assert res.get_json() == {"error": "메시지가 비어 있습니다."}
    assert agent.calls == []
    assert mcp_client.chat_history == []


def test_chat_empty_reply_uses_fallback(client, monkeypatch):
    monkeypatch.setattr(mcp_client, "run_agent", fake_agent(""))
    res = client.post("/chat", json={"message": "질문"})
    assert res.status_code == 200
    assert res.get_json()["reply"] == "응답이 생성되지 않았습니다."


# ---------------------------------------------------------------------------
# 실패 케이스
# ---------------------------------------------------------------------------
def test_make_server_unknown_transport_raises():
    with pytest.raises(ValueError, match="지원하지 않는 전송 방식"):
        mcp_client.make_server("sse")


def test_chat_without_json_body_rejected(client):
    res = client.post("/chat", data="hello", content_type="text/plain")
    assert res.status_code == 400


@pytest.mark.parametrize(("transport", "has_http_hint"), [("stdio", False), ("http", True)])
def test_chat_agent_error_returns_message(client, monkeypatch, transport, has_http_hint):
    monkeypatch.setattr(mcp_client, "TRANSPORT", transport)
    monkeypatch.setattr(mcp_client, "run_agent", fake_agent(error=ConnectionError("연결 거부")))

    res = client.post("/chat", json={"message": "질문"})
    body = res.get_json()

    assert res.status_code == 200
    assert body["tool_calls"] == []
    assert body["reply"].startswith("에러가 발생했습니다: 연결 거부")
    assert ("python mcp_server.py --http" in body["reply"]) is has_http_hint


def test_old_client_files_removed():
    for name in ("mcp_local_server.py", "mcp_1_stdio_client.py", "mcp_2_streamable_client.py", "mcp_1_index.html", "mcp_2_index.html"):
        assert not (MCP_DIR / name).exists(), name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "mcp_local_server" not in readme and "mcp_1_stdio_client" not in readme and "mcp_3_streamable_client" not in readme
