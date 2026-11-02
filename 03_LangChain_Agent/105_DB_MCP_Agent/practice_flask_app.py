"""
실습 문제 - 105. Chinook MCP + Flask Web UI

`# TODO` 표시된 부분을 채워 웹 챗봇을 완성하세요. (모범답안: `flask_app.py`)

실습 목표
  1. MCPAdapter로 stdio MCP 서버에 연결하고 도구 목록 가져오기
  2. MCP 도구 + InMemorySaver로 에이전트 생성
  3. thread_id config로 에이전트 비동기 실행 / 대화 초기화

화면 분할로 모범답안 보면서 실습하기 (VS Code)
  방법 1 (추천) - 비교 보기: 탐색기에서 모범답안 파일 클릭 -> Ctrl+클릭으로 이 파일 선택
                  -> 우클릭 [선택 항목 비교(Compare Selected)]
                  왼쪽 = 모범답안, 오른쪽 = 실습 파일(편집 가능), 채울 부분이 색으로 표시됨
  방법 2 - 좌우 분할: 이 파일 탭 우클릭 -> [오른쪽으로 분할(Split Right)]
                      -> 한쪽 창에서 Ctrl+P로 모범답안 파일 열기

LangChain v1 (create_agent + InMemorySaver) 기반 웹 챗봇.

실행 방법:
    python practice_flask_app.py
    브라우저에서 http://127.0.0.1:5000 접속
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

# 이 폴더의 절대 경로 (agent_server.py, index.html 위치 기준)
BASE_DIR = Path(__file__).resolve().parent

# Chinook MCP 서버 스크립트 경로 (직접 완성한 practice_agent_server.py)
_SERVER_SCRIPT = BASE_DIR / "practice_agent_server.py"

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
    """MCP 서버 연결 후 LangChain v1 에이전트 생성 (1회 실행)."""
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model
    from langchain.mcp import MCPAdapter
    from langgraph.checkpoint.memory import InMemorySaver

    if not _SERVER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Chinook MCP 서버를 찾을 수 없습니다: {_SERVER_SCRIPT}"
        )

    model = init_chat_model(
        os.environ["OPENAI_MODEL"],
        model_provider="openai",
        use_responses_api=True,        # Responses API 사용 (Chat Completions에서는 추론 모델의 도구 호출 불가)
        reasoning={"effort": "low"},   # 추론 강도 낮춤 → 실습 속도·비용 절감
    )

    # MCPConfig 형식(mcpServers)으로 서버 지정 — command 가 있으면 stdio 로 자동 연결
    adapter = MCPAdapter({
        "mcpServers": {
            "chinook": {
                # TODO: command는 이 앱과 같은 파이썬(sys.executable),
                #       args는 서버 스크립트 경로(_SERVER_SCRIPT)를 문자열로 담은 리스트로 지정하세요.
                "command": ...,
                "args": ...,
            }
        }
    })
    # 도구 목록 조회 → LangChain 도구로 변환 (반환된 도구는 이후에도 계속 호출 가능)
    # TODO: adapter에서 MCP 도구 목록을 비동기로 가져오세요. (await)
    tools = ...

    return create_agent(
        model=model,
        tools=...,         # TODO: MCP에서 가져온 도구 목록
        checkpointer=...,  # TODO: 메모리 기반 checkpointer (InMemorySaver)
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
    # TODO: thread_id를 담은 에이전트 호출용 config를 만드세요. ({"configurable": {...}})
    config = ...
    # TODO: 사용자 메시지와 config로 에이전트를 비동기 실행(ainvoke)하세요.
    response = ...
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
    # TODO: 세션의 thread_id를 새 값(uuid.uuid4().hex)으로 바꾸세요.
    ...
    return jsonify({"ok": True})


if __name__ == "__main__":
    # 개발용 서버 실행 (reloader는 백그라운드 루프 중복 생성 방지 위해 비활성화)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
