"""105_DB_MCP_Agent/agent_server.py (Chinook DB MCP 서버) 검증.

105·106 앱이 함께 사용하는 서버이므로, 도구·리소스 이름과 동작을 유지하면서
읽기 전용 연결 + SELECT 검증으로 쓰기 쿼리를 막는지 확인한다.
FastMCP 클라이언트를 서버 객체에 메모리 내 연결해서 호출한다. (LLM 호출 없음)
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "03_LangChain_Agent" / "105_DB_MCP_Agent" / "agent_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("agent_server_under_test", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = _load_server()


def run(coro_fn):
    """서버에 연결(lifespan 실행)한 상태로 coro_fn(client) 실행"""
    async def main():
        async with Client(server.mcp) as client:
            return await coro_fn(client)
    return asyncio.run(main())


def call(name: str, args: dict | None = None):
    async def go(client):
        result = await client.call_tool(name, args or {})
        if result.data is not None:
            return result.data
        return json.loads(result.content[0].text)  # 반환 타입이 그냥 list 이면 JSON 텍스트로 옴
    return run(go)


def artist_count() -> str:
    return call("execute_sql_query", {"query": "SELECT COUNT(*) FROM Artist"})


# ---------------------------------------------------------------------------
# 정상 케이스: 기존 도구·리소스 이름과 동작 유지 (105·106 앱 호환)
# ---------------------------------------------------------------------------
def test_tools_and_resources_unchanged():
    async def go(client):
        tools = sorted(t.name for t in await client.list_tools())
        resources = [str(r.uri) for r in await client.list_resources()]
        templates = [t.uri_template for t in await client.list_resource_templates()]
        return tools, resources, templates
    tools, resources, templates = run(go)
    assert tools == ["execute_sql_query", "get_table_schema", "list_tables", "validate_sql_query"]
    assert resources == ["database://info"] and templates == ["table://{table_name}"]


def test_no_langchain_dependency():
    # MCP 서버는 프레임워크와 무관해야 하므로 langchain 계열 패키지를 import 하지 않음
    imports = [line for line in SERVER.read_text(encoding="utf-8").splitlines()
               if line.startswith(("import ", "from "))]
    assert not [line for line in imports if "langchain" in line], imports


def test_select_query():
    assert artist_count() == "[(275,)]"


def test_list_tables():
    tables = call("list_tables")
    assert tables == sorted(tables) and {"Artist", "Album", "Track"} <= set(tables)
    assert not any(t.startswith("sqlite_") for t in tables)


def test_table_schema():
    out = call("get_table_schema", {"table_name": "Artist"})
    assert out.startswith("CREATE TABLE") and "3 rows from Artist table" in out and "AC/DC" in out


def test_validate_valid_select():
    assert call("validate_sql_query", {"query": "SELECT Name FROM Artist"})["valid"] is True


def test_resources():
    async def go(client):
        info = await client.read_resource("database://info")
        table = await client.read_resource("table://Album")
        return json.loads(info[0].text), table[0].text
    info, table = run(go)
    assert info["database"] == "Chinook" and info["tables_count"] == 11
    assert table.startswith("CREATE TABLE") and table.endswith("...") and len(table) == 103


# ---------------------------------------------------------------------------
# 경계값
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query",
    ["select count(*) from Artist;", "  WITH a AS (SELECT 1) SELECT * FROM a  ", "SELECT 1;   "],
)
def test_select_variants_allowed(query):
    assert not call("execute_sql_query", {"query": query}).startswith("쿼리 실행 중 오류")


def test_unknown_table_schema():
    out = call("get_table_schema", {"table_name": "NoSuchTable"})
    assert "스키마 조회 중 오류" in out and "'NoSuchTable' 테이블이 데이터베이스에 없습니다" in out


# ---------------------------------------------------------------------------
# 실패 케이스: 쓰기·다중 문장 차단, DB 내용 불변
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("DELETE FROM Artist WHERE ArtistId > 270", "SELECT(또는 WITH)"),
        ("UPDATE Artist SET Name = 'x'", "SELECT(또는 WITH)"),
        ("DROP TABLE Artist", "SELECT(또는 WITH)"),
        ("SELECT 1; DELETE FROM Artist", "여러 SQL 문장"),
        ("WITH a AS (SELECT 1) DELETE FROM Artist", "readonly"),  # 검증 통과 → 읽기 전용 연결이 차단
    ],
)
def test_write_queries_blocked(query, reason):
    out = call("execute_sql_query", {"query": query})
    assert out.startswith("쿼리 실행 중 오류") and reason in out, out
    assert artist_count() == "[(275,)]"


@pytest.mark.parametrize(
    "query",
    ["SELECT 1; DELETE FROM Artist", "DELETE FROM Artist", "SELEC Name FROM Artist"],
)
def test_validate_rejects_invalid_or_write(query):
    result = call("validate_sql_query", {"query": query})
    assert result["valid"] is False
    assert artist_count() == "[(275,)]"


def test_tools_fail_cleanly_before_server_start():
    # 서버 시작(lifespan) 전에는 db_path 가 없으므로 예외 대신 오류 메시지를 반환
    assert server.db_path is None
    assert "데이터베이스가 연결되지 않았습니다" in server.execute_sql_query("SELECT 1")
    assert "데이터베이스가 연결되지 않았습니다" in server.list_tables()[0]
