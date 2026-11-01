# ---------------------------------------------------------
# MCP 1: MCPServerStdio 클라이언트 (Flask)
# ---------------------------------------------------------
# mcp.json에 정의된 로컬 MCP 서버를 subprocess(stdio)로 연결.
# 문서 기능 반영: name, async with, tool_filter, mcp_config, cache_tools_list
#
# - Flask로 JSON API를 제공하고, HTML 한 장(mcp_1_index.html)으로 채팅 UI 구현
# - 실행: python mcp_1_stdio_client.py  ->  브라우저에서 http://localhost:8001 접속
#
# 라우트 구성:
#   GET  /       채팅 UI(mcp_1_index.html) 반환
#   POST /chat   {"message": "..."} 를 받아 에이전트 실행 후
#                {"reply": "...", "tool_calls": [...]} 반환
#   POST /reset  대화 기록 초기화
# ---------------------------------------------------------

import os
import sys
import asyncio
import json
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, create_static_tool_filter
from dotenv import load_dotenv

load_dotenv()

# Windows에서 asyncio subprocess 호환을 위한 이벤트 루프 정책
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

PORT = 8001
BASE_DIR = Path(__file__).resolve().parent


# ---------- 메인 비동기 로직 ----------
async def run_agent(messages: list[dict]) -> tuple[str, list[str]]:
    """async with 컨텍스트 매니저로 MCP 서버 연결 → 에이전트 실행.

    반환: (최종 답변, 사용한 도구 이름 목록)
    """

    # mcp.json 읽기
    with open(BASE_DIR / "mcp.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    server_config = config["mcpServers"]["mcp_local_server"]

    # ── MCPServerStdio: async with 패턴 ──
    async with MCPServerStdio(
        name="mcp_local_server",                         # name 파라미터
        params={
            "command": server_config["command"],
            "args": server_config.get("args", []),
            "cwd": str(BASE_DIR),                        # 서버 스크립트 위치 기준 실행
        },
        cache_tools_list=True,                            # 도구 목록 캐싱
        # ── tool_filter: 허용할 도구만 노출 ──
        tool_filter=create_static_tool_filter(
            allowed_tool_names=[
                "get_price", "add_item", "list_items",
                "list_tables", "get_table_schema", "execute_sql_query",
            ]
        ),
    ) as server:

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
            # ── mcp_config: 스키마 strict 변환 ──
            mcp_config={"convert_schemas_to_strict": True},
        )

        # ── 에이전트 실행 (완성된 답변을 한 번에 반환) ──
        result = await Runner.run(agent, input=messages)

        # 실행 중 호출된 도구 이름 수집 (Streamlit 버전의 st.toast에 해당)
        tool_calls = [
            item.raw_item.name
            for item in result.new_items
            if item.type == "tool_call_item"
        ]

    return str(result.final_output or ""), tool_calls


# ---------------------------------------------------------------------------------
# 대화 상태 (서버 메모리에 유지 - Streamlit session_state에 해당)
# - threaded=True로 실행하면 요청이 동시에 처리될 수 있으므로 Lock으로 보호
# ---------------------------------------------------------------------------------
chat_history = []
history_lock = threading.Lock()

# ---------------------------------------------------------------------------------
# Flask 앱과 라우트
# ---------------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    """채팅 UI 페이지 반환"""
    return send_file(BASE_DIR / "mcp_1_index.html")


@app.post("/chat")
def chat():
    """사용자 메시지를 받아 에이전트를 실행하고 답변을 JSON으로 반환"""
    body = request.get_json(silent=True) or {}
    user_input = (body.get("message") or "").strip()
    if not user_input:
        return jsonify(error="메시지가 비어 있습니다."), 400

    # 사용자 메시지를 대화 기록에 추가하고, 전체 기록의 복사본으로 에이전트 실행
    with history_lock:
        chat_history.append({"role": "user", "content": user_input})
        history = list(chat_history)

    # asyncio.run: Flask 요청 처리 스레드에는 실행 중인 이벤트 루프가 없으므로 사용 가능
    # (요청마다 MCP 서버를 subprocess로 새로 연결했다가 종료 - async with)
    try:
        reply, tool_calls = asyncio.run(run_agent(history))
        if not reply:
            reply = "응답이 생성되지 않았습니다."
    except Exception as e:
        reply, tool_calls = f"에러가 발생했습니다: {e}", []

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
    print(f"MCP stdio 챗봇 서버 시작: http://localhost:{PORT}  (종료: Ctrl+C)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
