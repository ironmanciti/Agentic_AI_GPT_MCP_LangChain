"""
쇼핑몰 고객 지원 RAG 챗봇 (Flask + HTML)

- Streamlit 없이 Flask로 JSON API를 제공하고, HTML 한 장(바닐라 JavaScript)으로 채팅 UI를 구현합니다.
- 에이전트 구성은 10_Chatbot_RAG_Agent.ipynb와 동일:
  웹 검색(WebSearchTool) + RAG(FileSearchTool + OpenAI Vector Store)
- 실행: python app.py  ->  브라우저에서 http://localhost:8000 접속

라우트 구성:
  GET  /       채팅 UI(index.html) 반환
  POST /chat   {"message": "..."} 를 받아 에이전트 실행 후 {"reply": "..."} 반환
  POST /reset  대화 기록 초기화
"""

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

import openai
from agents import Agent, Runner, WebSearchTool, FileSearchTool

load_dotenv()

OPENAI_MODEL = os.environ["OPENAI_MODEL"]
PORT = 8000
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------
# Vector Store 초기 설정 (서버 시작 시 1회)
# - 10/11번 예제와 동일: 같은 이름의 Vector Store가 있으면 재사용
# ---------------------------------------------------------------------------------
VS_NAME = "notebook-faq-kb"
DATA_DIR = BASE_DIR.parent / "data"

client = openai.OpenAI()
existing = [vs for vs in client.vector_stores.list() if vs.name == VS_NAME]

if existing:
    vs_id = existing[0].id
    print(f"기존 Vector Store 재사용: {vs_id}")
else:
    vs = client.vector_stores.create(name=VS_NAME)
    vs_id = vs.id
    files_to_upload = [
        open(DATA_DIR / "faqs.json", "rb"),
        open(DATA_DIR / "knowledgeBase.json", "rb"),
    ]
    try:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vs_id, files=files_to_upload,
        )
        print(f"Vector Store 생성 및 파일 업로드 완료: {vs_id}")
    finally:
        for f in files_to_upload:
            f.close()


# 에이전트는 서버 시작 시 1회만 생성해서 재사용
agent = Agent(
    name="쇼핑몰 고객 지원 에이전트",
    instructions=(
        "당신은 온라인 쇼핑몰의 고객 지원 상담원입니다. "
        "비밀번호·이메일 변경 등 계정 관리, 환불 정책, 주문 조회, 전화 상담 시간, "
        "개인정보 보안에 관한 질문은 반드시 file_search 도구로 지식 베이스를 먼저 검색하고, "
        "검색된 내용에 근거해서만 답하세요. "
        "지식 베이스에 없는 내용은 추측하지 말고 고객 지원팀 문의를 안내하세요. "
        "WebSearchTool은 지식 베이스와 무관한 최신 정보가 필요할 때만 사용하세요. "
        "답변은 친절하고 간결하게 하세요."
    ),
    model=OPENAI_MODEL,
    tools=[WebSearchTool(), FileSearchTool(vector_store_ids=[vs_id])],
)

# ---------------------------------------------------------------------------------
# 대화 상태 (서버 메모리에 유지 - Streamlit session_state에 해당)
# - 현재는 threaded=False라 요청이 하나씩 처리되지만, 멀티스레드로 바꿀 때를 대비해 Lock으로 보호
# ---------------------------------------------------------------------------------
messages = []
messages_lock = threading.Lock()

# ---------------------------------------------------------------------------------
# Flask 앱과 라우트
# ---------------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    """채팅 UI 페이지 반환"""
    return send_file(BASE_DIR / "index.html")


@app.post("/chat")
def chat():
    """사용자 메시지를 받아 에이전트를 실행하고 답변을 JSON으로 반환"""
    body = request.get_json(silent=True) or {}
    user_input = (body.get("message") or "").strip()
    if not user_input:
        return jsonify(error="메시지가 비어 있습니다."), 400

    # 사용자 메시지를 대화 기록에 추가하고, 전체 기록의 복사본으로 에이전트 실행
    with messages_lock:
        messages.append({"role": "user", "content": user_input})
        history = list(messages)

    # Runner.run_sync: 요청 처리 스레드에는 실행 중인 이벤트 루프가 없으므로 사용 가능
    try:
        result = Runner.run_sync(agent, input=history)
        reply = str(result.final_output or "응답이 생성되지 않았습니다.")
    except Exception as e:
        reply = f"에러가 발생했습니다: {e}"

    with messages_lock:
        messages.append({"role": "assistant", "content": reply})

    return jsonify(reply=reply)


@app.post("/reset")
def reset():
    """대화 기록 초기화"""
    with messages_lock:
        messages.clear()
    return jsonify(ok=True)


if __name__ == "__main__":
    print(f"RAG 챗봇 서버 시작: http://localhost:{PORT}  (종료: Ctrl+C)")
    # threaded=False: 모든 요청을 같은 스레드(같은 이벤트 루프)에서 처리한다.
    # threaded=True면 요청마다 스레드와 이벤트 루프가 달라져, 공유 OpenAI 클라이언트의
    # 연결이 다른 루프에서 재사용되면서 응답이 멈추는 문제가 생긴다.
    app.run(host="127.0.0.1", port=PORT, threaded=False)
