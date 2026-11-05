# ---------------------------------------------------------
# MCP 클라이언트 (Flask) - stdio / Streamable HTTP 겸용
# ---------------------------------------------------------
# 같은 MCP 서버(mcp_server.py)에 두 가지 전송 방식으로 연결한다.
#   - stdio : 클라이언트가 mcp.json의 명령으로 서버를 subprocess로 직접 기동
#   - http  : 미리 띄워 둔 HTTP 서버(http://localhost:8000/mcp)에 URL로 접속
# 문서 기능 반영: MCPServerStdio, MCPServerStreamableHttp, name, async with,
#                tool_filter, mcp_config, cache_tools_list
#
# - Flask로 JSON API를 제공하고, HTML 한 장(mcp_index.html)으로 채팅 UI 구현
# - 실행:
#     python mcp_client.py           (stdio)
#     python mcp_server.py --http   ->   python mcp_client.py --http   (HTTP)
#   브라우저에서 http://localhost:8001 접속
#
# 라우트 구성:
#   GET  /       채팅 UI(mcp_index.html) 반환
#   POST /chat   {"message": "..."} 를 받아 에이전트 실행 후
#                {"reply": "...", "tool_calls": [...]} 반환
#   POST /reset  대화 기록 초기화
# ---------------------------------------------------------

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import (
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request

load_dotenv()

# Windows에서 asyncio subprocess 호환을 위한 이벤트 루프 정책
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

PORT = 8001   # Flask 채팅 서버 포트 (브라우저에서 http://localhost:8001 접속)

# 이 파일이 있는 폴더 — mcp.json, mcp_index.html, mcp_server.py를 찾기 위한 기준 경로
BASE_DIR = Path(__file__).resolve().parent
HTTP_SERVER_URL = "http://localhost:8000/mcp"   # mcp_server.py --http 주소

# 전송 방식 → 화면 표시용 이름 
TRANSPORT_LABELS = {"stdio": "stdio", "http": "Streamable HTTP"}

# tool_filter로 에이전트에 노출할 도구 목록 
ALLOWED_TOOLS = [
    "get_price", "add_item", "list_items",                   # 의류 재고 도구
    "list_tables", "get_table_schema", "execute_sql_query",  # Chinook DB 도구
]

TRANSPORT = "stdio"

def parse_transport(argv: list[str]) -> str:
    """명령행 인자에 --http가 있으면 "http", 없으면 "stdio" """
    return "http" if "--http" in argv else "stdio"


def make_server(transport: str):
    """전송 방식에 맞는 MCP 서버 객체 생성 (연결은 async with에서 이루어짐)"""
    # ── tool_filter: 허용할 도구만 노출 / cache_tools_list: 도구 목록 캐싱 ──
    common = {
        "name": "mcp_server", 
        "cache_tools_list": True,
        "client_session_timeout_seconds": 30,   # 서버 기동이 느린 PC 대비 (기본 5초)
        "tool_filter": create_static_tool_filter(allowed_tool_names=ALLOWED_TOOLS),
    }

    if transport == "stdio":
        # mcp.json 읽기 → 서버를 subprocess로 기동
        with open(BASE_DIR / "mcp.json", "r", encoding="utf-8") as f:
            server_config = json.load(f)["mcpServers"]["mcp_server"]

        command = server_config["command"]
        # 지금 클라이언트를 실행 중인 인터프리터(sys.executable)로 서버 기동
        if command == "python":
            command = sys.executable
        return MCPServerStdio(
            params={
                "command": command,
                "args": server_config.get("args", []),
                "cwd": str(BASE_DIR),                    # 서버 스크립트 위치 기준 실행
            },
            **common,
        )

    if transport == "http":
        # 이미 실행 중인 HTTP 서버에 URL로 접속
        return MCPServerStreamableHttp(params={"url": HTTP_SERVER_URL}, **common)

    raise ValueError(f"지원하지 않는 전송 방식: {transport}")


# ---------- 메인 비동기 로직 ----------
async def run_agent(messages: list[dict], transport: str) -> tuple[str, list[str]]:
    """async with 컨텍스트 매니저로 MCP 서버 연결 → 에이전트 실행.

    반환: (최종 답변, 사용한 도구 이름 목록)
    """
    async with make_server(transport) as server:

        # ── Agent 생성 ──
        agent = Agent(
            name="Assistant",
            instructions=(
                "너는 의류 가격 조회·추가와 Chinook DB 쿼리를 도와주는 에이전트야. "
                "get_price, add_item, list_items, list_tables, get_table_schema, "
                "execute_sql_query 도구를 활용해 답해."
            ),
            model=os.environ["OPENAI_MODEL"],
            mcp_servers=[server],
            # OpenAI의 strict 모드 형식으로 변환 
            mcp_config={"convert_schemas_to_strict": True},
        )

        # ── 에이전트 실행 (완성된 답변을 한 번에 반환) ──
        result = await Runner.run(agent, input=messages)

        # 실행 중 호출된 도구 이름 수집 → 채팅 화면에 "도구 활용" 말풍선으로 표시
        tool_calls = [
            item.raw_item.name 
            for item in result.new_items if item.type == "tool_call_item"
        ]

    return str(result.final_output or ""), tool_calls


# ---------------------------------------------------------------------------------
# 대화 상태 (Flask 서버 메모리에 유지 - 모든 브라우저가 하나의 대화를 공유)
# - threaded=True로 실행하면 요청이 동시에 처리될 수 있으므로 Lock으로 보호
# ---------------------------------------------------------------------------------
chat_history = []
history_lock = threading.Lock()

# ---------------------------------------------------------------------------------
# 에이전트 전용 이벤트 루프 (백그라운드 스레드에서 계속 실행)
# - SDK의 공유 HTTP 클라이언트는 연결을 처음 만든 루프에 묶이므로,
#   요청마다 asyncio.run으로 루프를 새로 만들고 닫으면 "Event loop is closed" 발생
# ---------------------------------------------------------------------------------
agent_loop = asyncio.new_event_loop()
threading.Thread(target=agent_loop.run_forever, daemon=True).start()

# ---------------------------------------------------------------------------------
# Flask 앱과 라우트
# ---------------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    """채팅 UI 페이지 반환 (제목·안내 문구에 전송 방식 표시)"""
    html = (BASE_DIR / "mcp_index.html").read_text(encoding="utf-8")
    return render_template_string(
        html, transport=TRANSPORT, transport_label=TRANSPORT_LABELS[TRANSPORT]
    )


@app.post("/chat")
def chat():
    """사용자 메시지를 받아 에이전트를 실행하고 답변을 JSON으로 반환"""
    # 요청 본문(JSON) 파싱 — silent=True: JSON이 아니거나 깨져도 예외 대신 None 반환 → 빈 dict로 대체
    body = request.get_json(silent=True) or {}
    # "message" 키가 없거나 null이어도 빈 문자열로 처리, 앞뒤 공백 제거 (공백만 입력 → 빈 메시지)
    user_input = (body.get("message") or "").strip()
    
    if not user_input:
        return jsonify(error="메시지가 비어 있습니다."), 400

    # 사용자 메시지를 대화 기록에 추가하고, 전체 기록의 복사본으로 에이전트 실행
    with history_lock:
        chat_history.append({"role": "user", "content": user_input})
        history = list(chat_history)

    # 전용 이벤트 루프에 코루틴을 맡기고, Flask 요청 스레드는 결과가 나올 때까지 대기
    # (요청마다 MCP 서버에 새로 연결했다가 종료 - async with)
    try:
        future = asyncio.run_coroutine_threadsafe(run_agent(history, TRANSPORT), agent_loop)
        reply, tool_calls = future.result()
        if not reply:
            reply = "응답이 생성되지 않았습니다."
    except Exception as e:
        reply, tool_calls = f"에러가 발생했습니다: {e}", []
        if TRANSPORT == "http":
            reply += "\nMCP 서버가 실행 중인지 확인하세요: python mcp_server.py --http"

    with history_lock:
        chat_history.append({"role": "assistant", "content": reply})

    return jsonify(reply=reply, tool_calls=tool_calls)


@app.post("/reset")
def reset():
    """대화 기록 초기화"""
    with history_lock:
        chat_history.clear()
    return jsonify(ok=True)


if __name__ == "__main__":
    TRANSPORT = parse_transport(sys.argv[1:])
    print(f"MCP 챗봇 서버 시작 ({TRANSPORT_LABELS[TRANSPORT]}): http://localhost:{PORT}  (종료: Ctrl+C)")
    if TRANSPORT == "http":
        print("먼저 MCP 서버를 띄워야 합니다:  python mcp_server.py --http")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
