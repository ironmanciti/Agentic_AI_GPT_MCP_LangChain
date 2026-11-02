"""
실습 문제 - 106. Chinook MCP + Notion MCP 통합 Web UI

`# TODO` 표시된 부분을 채워 통합 에이전트를 완성하세요. (모범답안: `flask_app.py`)
105 실습과 같은 부분(이벤트 루프, 에이전트 실행, 대화 초기화)은 완성된 코드로 제공합니다.

실습 목표
  1. .env의 Notion 토큰 읽기
  2. MCPAdapter 서버 목록에 Notion 공식 MCP 서버(npx) 추가
  3. 기본 Notion 페이지 ID를 시스템 프롬프트로 주입
  4. 시스템 프롬프트를 넣어 통합 에이전트 생성

화면 분할로 모범답안 보면서 실습하기 (VS Code)
  방법 1 (추천) - 비교 보기: 탐색기에서 모범답안 파일 클릭 -> Ctrl+클릭으로 이 파일 선택
                  -> 우클릭 [선택 항목 비교(Compare Selected)]
                  왼쪽 = 모범답안, 오른쪽 = 실습 파일(편집 가능), 채울 부분이 색으로 표시됨
  방법 2 - 좌우 분할: 이 파일 탭 우클릭 -> [오른쪽으로 분할(Split Right)]
                      -> 한쪽 창에서 Ctrl+P로 모범답안 파일 열기


105_DB_MCP_Agent 의 flask_app.py 에 이어지는 실습 과제입니다.
105 에서 만든 Chinook MCP 서버(agent_server.py)는 그대로 재사용하고,
MCPAdapter 의 서버 목록(mcpServers)에 Notion 공식 MCP 서버 항목 하나만
추가하면 DB 조회 + Notion 기록이 모두 가능한 통합 Agent 가 됩니다.
(서버가 2개 이상이면 도구 이름 앞에 서버 이름이 붙습니다: chinook_..., notion_...)

사전 준비 (.env):
    OPENAI_API_KEY=sk-...
    NOTION_API_KEY=ntn_...   (Notion 내부 Integration 토큰)
    NOTION_PAGE_ID=...       (기본 작업 페이지 ID, 선택)

실행 방법:
    python practice_flask_app.py
    브라우저에서 http://127.0.0.1:5001 접속
    (105 앱과 동시에 띄울 수 있도록 포트 5001 사용)

활용 예:
    "매출 상위 5개 국가를 조회해서 Notion 페이지로 정리해줘"
"""
import asyncio
import os
import sys
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file, session

load_dotenv()

# 이 폴더의 절대 경로 (index.html 위치 기준)
BASE_DIR = Path(__file__).resolve().parent

# Chinook MCP 서버 스크립트 경로: 105_DB_MCP_Agent 의 agent_server.py 재사용
_SERVER_SCRIPT = BASE_DIR.parent / "105_DB_MCP_Agent" / "agent_server.py"

# --------------------------------------------------------
# 백그라운드 이벤트 루프
# Flask는 동기 프레임워크이므로, 비동기 에이전트 호출을 위해
# 별도 스레드에서 전용 asyncio 이벤트 루프를 상시 실행한다.
# --------------------------------------------------------
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()

# 에이전트는 최초 요청 시 1회만 생성해 전역으로 재사용
# (InMemorySaver가 유지되어야 thread_id 기반 대화 기억이 동작)
_agent = None
_agent_lock = threading.Lock()


async def _build_agent():
    """Chinook + Notion MCP 서버 연결 후 LangChain v1 에이전트 생성 (1회 실행)."""
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain.mcp import MCPAdapter
    from langgraph.checkpoint.memory import InMemorySaver
    from mcp.client.stdio import get_default_environment

    if not _SERVER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Chinook MCP 서버를 찾을 수 없습니다: {_SERVER_SCRIPT}"
        )

    # TODO: .env의 NOTION_API_KEY 값을 읽으세요. (없으면 빈 문자열)
    notion_api_key = ...
    if not notion_api_key:
        raise ValueError(
            ".env 파일에 NOTION_API_KEY 가 설정되지 않았습니다. "
            "Notion 내부 Integration 토큰(ntn_...)을 설정하세요."
        )

    model = init_chat_model(
        os.environ["OPENAI_MODEL"],
        model_provider="openai",
        use_responses_api=True,        # Responses API 사용 (Chat Completions에서는 추론 모델의 도구 호출 불가)
        reasoning={"effort": "low"},   # 추론 강도 낮춤 → 실습 속도·비용 절감
    )

    # 105 와 동일한 구조에서 "notion" 서버 항목 하나만 추가
    adapter = MCPAdapter({
        "mcpServers": {
            "chinook": {
                "command": sys.executable,   # 이 앱과 같은 파이썬(가상환경)으로 서버 실행
                "args": [str(_SERVER_SCRIPT)],
            },
            "notion": {
                # TODO: npx로 Notion 공식 MCP 서버(@notionhq/notion-mcp-server)를 실행하도록
                #       command와 args를 지정하세요. (args에 "-y" 포함)
                "command": ...,
                "args": ...,
                # TODO: 기본 환경변수(get_default_environment())에 NOTION_TOKEN(notion_api_key)을 추가하세요.
                "env": ...,
            },
        }
    })
    # 두 서버의 도구를 한 번에 조회 → LangChain 도구로 변환
    tools = await adapter.list_tools()
    print(f"통합 도구 로딩 완료: {len(tools)}개 (Chinook + Notion)")

    # Notion MCP 에는 기본 페이지 개념이 없으므로, .env 의 기본 페이지 ID를
    # 시스템 프롬프트로 주입해 페이지 생성/수정 시 기준으로 사용하게 한다.
    default_page_id = os.getenv("NOTION_PAGE_ID", "")
    system_prompt = (
        "너는 Chinook 데이터베이스 조회와 Notion 페이지 작업을 도와주는 어시스턴트다. "
        "DB 관련 질문은 Chinook 도구를, Notion 작업은 Notion 도구를 사용하라."
    )
    if default_page_id:
        # TODO: "별도 지정이 없으면 기본 부모 페이지 ID {default_page_id}를 사용하라"는
        #       지시문을 system_prompt 뒤에 이어 붙이세요.
        system_prompt += ...

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=...,  # TODO: 위에서 만든 시스템 프롬프트
        checkpointer=InMemorySaver(),
    )


def _get_agent():
    """에이전트 싱글턴 반환 (없으면 백그라운드 루프에서 생성)."""
    global _agent
    with _agent_lock:
        if _agent is None:
            future = asyncio.run_coroutine_threadsafe(_build_agent(), _loop)
            _agent = future.result(timeout=120)
    return _agent


def _extract_text(message) -> str:
    """마지막 메시지에서 텍스트 추출 (content 블록 리스트 형태 대비)."""
    if message is None:
        return "(응답 없음)"
    # langchain-core v1: AIMessage.text 프로퍼티 (구버전은 메서드)
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    if text:
        return text
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        content = "".join(parts)
    return content or "(응답 없음)"


async def _ask_agent(agent, user_message: str, thread_id: str) -> str:
    """에이전트 비동기 호출 후 최종 응답 텍스트 반환."""
    config = {"configurable": {"thread_id": thread_id}}
    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    messages = response.get("messages", [])
    return _extract_text(messages[-1] if messages else None)


# --------------------------------------------------------
# Flask 앱
# --------------------------------------------------------
app = Flask(__name__)
# 브라우저 세션별 thread_id 저장용 시크릿 키
app.secret_key = os.environ.get("FLASK_SECRET_KEY", uuid.uuid4().hex)


@app.route("/")
def index():
    """채팅 UI 페이지 반환. 브라우저 세션마다 고유 thread_id 부여."""
    if "thread_id" not in session:
        session["thread_id"] = uuid.uuid4().hex
    return send_file(BASE_DIR / "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """사용자 질문을 받아 에이전트 응답을 JSON으로 반환."""
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "메시지가 비어 있습니다."}), 400

    thread_id = session.setdefault("thread_id", uuid.uuid4().hex)

    try:
        agent = _get_agent()
        future = asyncio.run_coroutine_threadsafe(
            _ask_agent(agent, user_message, thread_id), _loop
        )
        reply = future.result(timeout=300)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": f"에이전트 실행 중 오류: {e}"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """새 thread_id 발급으로 대화 기억 초기화."""
    session["thread_id"] = uuid.uuid4().hex
    return jsonify({"ok": True})


if __name__ == "__main__":
    # 개발용 서버 실행 (reloader는 백그라운드 루프 중복 생성 방지 위해 비활성화)
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
