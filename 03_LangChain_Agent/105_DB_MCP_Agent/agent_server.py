"""
Chinook DB MCP 서버 (105_DB_MCP_Agent, 106_Notion_DB_MCP_Total 에서 함께 사용)

SQL 에이전트에 필요한 DB 기능을 MCP 서버로 분리해, LangChain 뿐 아니라
Claude Desktop·Cursor·OpenAI Agents SDK 등 MCP 를 지원하는 어떤 클라이언트에서도
같은 도구를 재사용할 수 있게 한다. (같은 DB 기능을 노트북 안의 @tool 로 만든 예: 104_HITL_Agent)

[도구 4개 — 모델이 골라 호출]
  list_tables         테이블 목록 조회 (에이전트가 가장 먼저 호출해 어떤 데이터가 있는지 파악)
  get_table_schema    테이블 스키마(CREATE 문) + 샘플 3행 → 컬럼 이름·타입을 보고 정확한 SQL 작성
  validate_sql_query  실행하지 않고 문법만 검증 (SQLite 의 EXPLAIN QUERY PLAN 사용)
  execute_sql_query   조회 쿼리 실행 후 결과 반환

[리소스 2개 — 앱이 골라 읽음 (읽기 전용 컨텍스트)]
  database://info         DB 이름·테이블 목록 요약
  table://{table_name}    테이블 정보 미리보기 (URI 템플릿)

[안전장치 — 모델이 만든 SQL 을 그대로 믿지 않는다 (LangChain 공식 문서의 보안 권고)]
  1차: validate_select()   SELECT/WITH 로 시작하는 단일 문장만 허용 (쓰기·다중 문장 차단)
  2차: connect_readonly()  DB 를 읽기 전용(mode=ro)으로 연결 → 1차를 우회해도 데이터 변경 불가
  ※ 쿼리 점검을 LLM 에 맡기지 않는다: 서버는 모델 없이도 동작해야 하고,
     문법 확인은 DB 자체(EXPLAIN)가 더 정확하기 때문이다.
"""

from fastmcp import FastMCP
from contextlib import asynccontextmanager
import os
import sqlite3
import sys
import asyncio

# 데이터베이스 파일 경로 (서버 시작 시 lifespan 에서 확인 후 설정)
# sqlite3 연결은 스레드 간에 공유하면 안 되므로, 도구 호출마다 새로 열고 닫는다.
db_path = None


def connect_readonly() -> sqlite3.Connection:
    """DB를 읽기 전용(mode=ro)으로 연결 — 에이전트가 만든 SQL이 데이터를 바꾸지 못하도록 권한을 최소로 제한"""
    if db_path is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def validate_select(query: str) -> str:
    """조회(SELECT/WITH) 단일 문장만 허용하고, 아니면 ValueError 발생
    (WITH 로 시작하는 쓰기 문장은 읽기 전용 연결이 2차로 막음)"""
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise ValueError("여러 SQL 문장은 한 번에 실행할 수 없습니다.")
    if not q.lower().startswith(("select", "with")):
        raise ValueError("SELECT(또는 WITH) 조회 쿼리만 실행할 수 있습니다.")
    return q


def _table_names(con: sqlite3.Connection) -> list[str]:
    """내부 테이블(sqlite_...)을 제외한 테이블 이름 목록 (이름순)"""
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    ).fetchall()
    return [row[0] for row in rows]


def _table_info(con: sqlite3.Connection, table_name: str) -> str:
    """테이블 스키마(CREATE 문) + 샘플 3행"""
    if table_name not in _table_names(con):
        raise ValueError(f"'{table_name}' 테이블이 데이터베이스에 없습니다.")
    (create_sql,) = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
    ).fetchone()
    # table_name 은 위에서 실제 테이블 이름인지 확인했으므로 쿼리에 넣어도 안전
    cursor = con.execute(f'SELECT * FROM "{table_name}" LIMIT 3;')
    columns = "\t".join(col[0] for col in cursor.description)
    rows = "\n".join("\t".join(str(v) for v in row) for row in cursor.fetchall())
    return f"{create_sql}\n\n/*\n3 rows from {table_name} table:\n{columns}\n{rows}\n*/"


@asynccontextmanager
async def lifespan(app):
    """
    서버 시작/종료 시 데이터베이스 경로 확인·정리
    """
    global db_path
    try:
        # 서버 시작 시 데이터베이스 경로 계산 (절대경로)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "Chinook.db")

        # 데이터베이스 파일 존재 여부 확인
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chinook.db 파일을 찾을 수 없습니다: {path}")

        # 읽기 전용으로 열리는지 미리 확인
        db_path = path
        connect_readonly().close()
        print("Chinook 데이터베이스 연결 성공 (읽기 전용)", file=sys.stderr)

        # yield로 서버가 실행되는 동안 대기
        yield
    finally:
        # 서버 종료 시 정리
        if db_path:
            db_path = None
            print("Chinook 데이터베이스 연결 종료", file=sys.stderr)

# FastMCP 서버 인스턴스 생성
# - "ChinookDBAnalysis": MCP 서버의 이름
# - lifespan: 서버 시작/종료 시 실행될 함수
mcp = FastMCP("ChinookDBAnalysis", lifespan=lifespan)

@mcp.tool()
def execute_sql_query(query: str) -> str:
    """
    SQL 조회 쿼리를 실행하고 결과를 반환합니다. (SELECT 문만 허용)
    """
    try:
        q = validate_select(query)          # 1차 방어: 조회 쿼리만 허용
        con = connect_readonly()            # 2차 방어: 읽기 전용 연결
        try:
            return str(con.execute(q).fetchall())
        finally:
            con.close()
    except Exception as e:
        return f"쿼리 실행 중 오류 발생: {str(e)}"

@mcp.tool()
def get_table_schema(table_name: str) -> str:
    """
    특정 테이블의 컬럼명, 데이터 타입, 제약조건 등의
    스키마 정보를 조회합니다.
    """
    try:
        con = connect_readonly()
        try:
            return _table_info(con, table_name)
        finally:
            con.close()
    except Exception as e:
        return f"스키마 조회 중 오류 발생: {str(e)}"

@mcp.tool()
def list_tables() -> list:
    """
    현재 연결된 데이터베이스에서 사용 가능한
    모든 테이블의 이름을 리스트로 반환합니다.
    """
    try:
        con = connect_readonly()
        try:
            return _table_names(con)
        finally:
            con.close()
    except Exception as e:
        return [f"테이블 목록 조회 중 오류 발생: {str(e)}"]

@mcp.tool()
def validate_sql_query(query: str) -> dict:
    """
    실제로 쿼리를 실행하지 않고 문법만 검증하여
    SQL 쿼리가 올바른지 확인합니다.
    """
    try:
        q = validate_select(query)  # 조회 쿼리 단일 문장인지 먼저 확인
        con = connect_readonly()
        try:
            # 쿼리가 유효하면 실행 계획만 반환되고 데이터는 변경되지 않음
            con.execute(f"EXPLAIN QUERY PLAN {q}")
        finally:
            con.close()
        return {"valid": True, "message": "쿼리 문법이 올바릅니다."}
    except Exception as e:
        return {"valid": False, "message": f"쿼리 문법 오류: {str(e)}"}

@mcp.resource("database://info")
def get_database_info() -> dict:
    """
    Chinook 데이터베이스의 전반적인 정보를 반환합니다.
    MCP 리소스로 등록되어 클라이언트가 데이터베이스의
    기본 정보를 조회할 수 있습니다.

    Returns:
        dict: 데이터베이스 정보
              - database: 데이터베이스 이름
              - tables_count: 테이블 개수
              - tables: 테이블 목록
              - description: 데이터베이스 설명
    """
    try:
        # 모든 테이블 목록 조회
        con = connect_readonly()
        try:
            tables = _table_names(con)
        finally:
            con.close()
        return {
            "database": "Chinook",
            "tables_count": len(tables),
            "tables": tables,
            "description": "디지털 미디어 스토어 샘플 데이터베이스"
        }
    except Exception as e:
        return {"error": f"데이터베이스 정보 조회 중 오류: {str(e)}"}

@mcp.resource("table://{table_name}")
def get_table_info(table_name: str) -> str:
    """  
    MCP 리소스로 등록되어 동적 URI 패턴으로
    특정 테이블의 상세 정보를 반환합니다.    
    """
    try:
        # 테이블 스키마 정보 조회
        con = connect_readonly()
        try:
            schema_info = _table_info(con, table_name)
        finally:
            con.close()

        # 스키마 정보가 너무 길 경우 100자로 제한하고 "..." 추가
        if len(schema_info) > 100:
            return schema_info[:100] + "..."
        return schema_info
    except Exception as e:
        return f"테이블 정보 조회 중 오류: {str(e)}"

if __name__ == "__main__":
    try:
        print("MCP Server is running...", file=sys.stderr)
        mcp.run()
    except KeyboardInterrupt:
        # Ctrl+C로 서버 종료 시 정상 종료 메시지 출력
        print("\n✅ 서버가 정상적으로 종료되었습니다.", file=sys.stderr)
    except Exception as e:
        # 클라이언트가 정상적으로 연결을 끊을 때 anyio/asyncio 취소가 전파되며
        # BaseExceptionGroup("unhandled errors in a TaskGroup ...") 또는
        # asyncio.CancelledError/GeneratorExit 등이 발생할 수 있음. 이를 정상 종료로 간주.
        def _is_normal_disconnect(exc: BaseException) -> bool:
            # BaseExceptionGroup일 경우 내부 예외들을 검사
            if hasattr(exc, 'exceptions') and isinstance(exc.exceptions, (list, tuple)):
                return all(_is_normal_disconnect(sub) for sub in exc.exceptions) or False
            # anyio 취소 스코프 RuntimeError도 정상 종료로 간주
            if isinstance(exc, RuntimeError) and 'cancel scope' in str(exc):
                return True
            # Python 3.11+의 BaseExceptionGroup 중 TaskGroup 종료 메시지 포함 시 정상 종료 간주
            if "TaskGroup" in str(type(exc)) or "TaskGroup" in str(exc):
                return True
            return isinstance(exc, (asyncio.CancelledError, GeneratorExit, BrokenPipeError))

        if _is_normal_disconnect(e):
            print("\n✅ 클라이언트 연결 종료로 서버가 정상 종료되었습니다.", file=sys.stderr)
        else:
            # 예상치 못한 오류만 에러로 출력
            print(f"❌ 서버 오류: {e}", file=sys.stderr)