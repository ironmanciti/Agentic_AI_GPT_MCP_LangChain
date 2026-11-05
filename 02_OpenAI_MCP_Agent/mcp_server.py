# ---------------------------------------------------------
# FastMCP 통합 서버: 의류 가격 + Chinook DB (mcp_server)
# ---------------------------------------------------------
# 의류: get_price, add_item, list_items
# Chinook: execute_sql_query, list_tables, get_table_schema
# 전송: stdio (mcp.json + MCPServerStdio로 클라이언트가 자동 기동).
# ---------------------------------------------------------

from typing import Dict, List, Tuple
from contextlib import asynccontextmanager
import os
import sys
import sqlite3
from fastmcp import FastMCP

# ---------- 의류 재고 ----------
INVENTORY: Dict[str, float] = {
    "t-shirt": 19.99,
    "jeans": 59.90,
    "hoodie": 39.95,
}


# 품목명을 INVENTORY 키 형식으로 통일 (앞뒤 공백 제거 + 소문자)
def _normalize(item: str) -> str:
    return item.strip().lower()


# ---------- Chinook DB 전역 연결 ----------
db_conn = None

# @asynccontextmanager: yield 앞 = 시작 시 1회 실행, yield 뒤(finally) = 종료 시 1회 실행
# → FastMCP(lifespan=...)에 넘기면 서버 시작 시 DB 연결, 서버 종료 시 DB 연결 해제
@asynccontextmanager
async def lifespan(app):
    global db_conn
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "Chinook.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Chinook.db 파일을 찾을 수 없습니다: {db_path}")
        # check_same_thread=False: 도구 함수는 연결을 만든 스레드와 다른 스레드에서 실행됨
        db_conn = sqlite3.connect(db_path, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row
        print("Chinook 데이터베이스 연결 성공", file=sys.stderr)
        yield   # 여기서 멈춰 있는 동안 서버가 동작하며 도구 요청을 처리
    finally:    # 정상 종료든 에러든 항상 실행 → 연결 누수 방지
        if db_conn:
            db_conn.close()
            db_conn = None
            print("Chinook 데이터베이스 연결 종료", file=sys.stderr)


mcp = FastMCP("mcp_server", lifespan=lifespan)

# ---------- 의류 도구 ----------
@mcp.tool(description="의류 품목의 가격을 조회합니다. 항상 (found, price)를 반환합니다.")
def get_price(item: str) -> Tuple[bool, float]:
    key = _normalize(item)
    return (key in INVENTORY, INVENTORY.get(key, 0.0))


@mcp.tool(description="의류 품목을 추가하거나 가격을 업데이트합니다. 항상 (item, price)를 반환합니다.")
def add_item(item: str, price: float) -> Tuple[str, float]:
    key = _normalize(item)
    INVENTORY[key] = max(price, 0.0)
    return key, INVENTORY[key]


@mcp.tool(description="모든 의류 품목과 가격 목록을 반환합니다.")
def list_items() -> List[Tuple[str, float]]:
    return sorted(INVENTORY.items())


# ---------- Chinook DB 도구 ----------
@mcp.tool(description="SQL 쿼리를 실행하고 결과를 반환합니다.")
def execute_sql_query(query: str) -> str:
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    try:
        cursor = db_conn.cursor()
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            if not rows:
                return "쿼리 결과가 없습니다."
            columns = [d[0] for d in cursor.description]
            lines = [" | ".join(columns), "-" * (len(" | ".join(columns)))]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)
        else:
            db_conn.commit()
            return f"쿼리가 성공적으로 실행되었습니다. 영향받은 행: {cursor.rowcount}"
    except Exception as e:
        return f"쿼리 실행 중 오류 발생: {str(e)}"


@mcp.tool(description="Chinook 데이터베이스의 모든 테이블 목록을 반환합니다.")
def list_tables() -> list:
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        return [f"테이블 목록 조회 중 오류 발생: {str(e)}"]


@mcp.tool(description="특정 테이블의 스키마 정보(컬럼명, 데이터 타입 등)를 조회합니다.")
def get_table_schema(table_name: str) -> str:
    if db_conn is None:
        raise ValueError("데이터베이스가 연결되지 않았습니다.")
    try:
        cursor = db_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        if not columns:
            return f"테이블 '{table_name}'을 찾을 수 없습니다."
        lines = [f"테이블: {table_name}", "-" * 40, "컬럼명 | 타입 | NULL 허용 | 기본값", "-" * 40]
        for col in columns:
            lines.append(f"{col[1]} | {col[2]} | {'NO' if col[3] else 'YES'} | {col[4] or ''}")
        return "\n".join(lines)
    except Exception as e:
        return f"스키마 조회 중 오류 발생: {str(e)}"


# ---------- 서버 실행 ----------
# 기본: stdio (mcp_client.py가 subprocess로 기동)
# --http 옵션: Streamable HTTP (python mcp_client.py --http 용)
#   python mcp_server.py --http
if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
    else:
        mcp.run()
