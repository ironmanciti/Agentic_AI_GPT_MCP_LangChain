# ---------------------------------------------------------
# MCP 2: MCPServerStreamableHttp + MCPServerManager (Flask)
# ---------------------------------------------------------
# HTTP 기반 원격 MCP 서버에 연결하고, MCPServerManager로 다중 서버를
# 통합 관리. 연결 실패 처리, 서버 상태 모니터링 포함.
# 문서 기능 반영: MCPServerStreamableHttp, MCPServerManager, mcp_config
#
# - Flask로 JSON API를 제공하고, HTML 한 장(mcp_2_index.html)으로 채팅 UI 구현
# - 실행 전 HTTP MCP 서버가 필요합니다:
#     python mcp_local_server.py --http        (port 8000)
# - 실행: python mcp_2_streamable_client.py  ->  http://localhost:8002 접속
#
# 라우트 구성:
#   GET  /       채팅 UI(mcp_2_index.html) 반환
#   POST /chat   {"message": "...", "server_urls": [...]} 를 받아 에이전트 실행 후
#                {"reply": "...", "tool_calls": [...], "active": n, "failed": n} 반환
#   POST /reset  대화 기록 초기화
# ---------------------------------------------------------

import sys
import asyncio
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp, MCPServerManager
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

PORT = 8002
BASE_DIR = Path(__file__).resolve().parent


# ---------- 메인 비동기 로직 ----------
async def run_agent(
    messages: list[dict],
    server_urls: list[str],
) -> tuple[str, list[str], int, int]:
    """MCPServerStreamableHttp + MCPServerManager로 다중 서버 연결 후 에이전트 실행.

    반환: (최종 답변, 사용한 도구 이름 목록, 활성 서버 수, 실패 서버 수)
    """

    # ── 서버 인스턴스 생성 ──
    servers = []
    for i, url in enumerate(server_urls):
        server = MCPServerStreamableHttp(
            name=f"MCP Server {i + 1}",
            params={"url": url},
            cache_tools_list=True,
        )
        servers.append(server)

    # ── MCPServerManager: 다중 서버 통합 관리 ──
    async with MCPServerManager(servers) as manager:
        active_count = len(manager.active_servers)
        failed_count = len(manager.failed_servers)

        if active_count == 0:
            return "연결된 MCP 서버가 없습니다. 서버 URL을 확인해주세요.", [], 0, failed_count

        # ── Agent 생성 ──
        agent = Agent(
            name="Multi-Server Assistant",
            instructions=(
                "너는 여러 MCP 서버의 도구를 활용하는 에이전트야. "
                "연결된 서버들의 도구를 사용해서 사용자 질문에 답해줘. "
                "한국어로 답변해."
            ),
            model="gpt-4o-mini",
            mcp_servers=manager.active_servers,
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

    return str(result.final_output or ""), tool_calls, active_count, failed_count


# ---------------------------------------------------------------------------------
# 대화 상태 (서버 메모리에 유지 - Streamlit session_state에 해당)
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
    return send_file(BASE_DIR / "mcp_2_index.html")


@app.post("/chat")
def chat():
    """사용자 메시지 + 서버 URL 목록을 받아 에이전트를 실행하고 답변을 JSON으로 반환"""
    body = request.get_json(silent=True) or {}
    user_input = (body.get("message") or "").strip()
    server_urls = [u.strip() for u in (body.get("server_urls") or []) if u.strip()]

    if not user_input:
        return jsonify(error="메시지가 비어 있습니다."), 400
    if not server_urls:
        return jsonify(error="서버 URL을 최소 1개 입력해주세요."), 400

    with history_lock:
        chat_history.append({"role": "user", "content": user_input})
        history = list(chat_history)

    # asyncio.run: Flask 요청 처리 스레드에는 실행 중인 이벤트 루프가 없으므로 사용 가능
    # (요청마다 MCPServerManager로 서버들에 연결했다가 종료 - async with)
    try:
        reply, tool_calls, active, failed = asyncio.run(run_agent(history, server_urls))
        if not reply:
            reply = "응답이 생성되지 않았습니다."
    except Exception as e:
        reply, tool_calls, active, failed = f"에러가 발생했습니다: {e}", [], 0, len(server_urls)

    with history_lock:
        chat_history.append({"role": "assistant", "content": reply})

    return jsonify(reply=reply, tool_calls=tool_calls, active=active, failed=failed)


@app.post("/reset")
def reset():
    """대화 기록 초기화"""
    with history_lock:
        chat_history.clear()
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"MCP Streamable HTTP 챗봇 서버 시작: http://localhost:{PORT}  (종료: Ctrl+C)")
    print("먼저 MCP 서버를 띄워야 합니다:  python mcp_local_server.py --http")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
