"""
LangChain ReAct Agent 기반 Chatbot (Flask + HTML)

- 에이전트 구성은 101_Tools_Agents + 102_Memory_Concepts 노트북 내용을 반영:
  - 고객 DB 검색(search_db) + 현재 기온 조회(get_weather) 도구를 사용하는 ReAct Agent
  - InMemorySaver(checkpointer) 기반 단기 메모리로 대화 이력을 자동 관리
- 실행: python 101_Tools_Memory_Agent.py  ->  브라우저에서 http://localhost:8000 접속

라우트 구성:
  GET  /         채팅 UI(101_Tools_Memory_Agent.html) 반환
  POST /chat     {"message": "..."} 를 받아 에이전트 실행 후 {"reply": "..."} 반환
  POST /reset    대화 기록 초기화 (새 thread_id 발급)
  POST /summary  지금까지의 대화 내용을 LLM으로 요약하여 반환
"""

import threading
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL_NAME = "gpt-5.4-mini"
PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT = "당신은 도움이 되는 어시스턴트입니다. 주어진 도구를 이용해 답변하세요."

# ---------------------------------------------------------------------------------
# 도구 정의 (101_Tools_Agents 노트북과 동일)
# ---------------------------------------------------------------------------------
@tool
def search_db(query: str, limit: int = 10) -> str:
    """검색어(query)에 해당하는 고객 데이터베이스 레코드를 조회합니다.

    Args:
        query: 검색할 키워드 또는 문장
        limit: 반환할 최대 결과 개수
    """
    return f"'{query}'에 대한 검색 결과 {limit}개를 찾았습니다."


class WeatherInput(BaseModel):
    """날씨 질의에 사용할 입력 스키마"""
    latitude: float = Field(description="질의할 지역의 위도를 입력합니다.")
    longitude: float = Field(description="질의할 지역의 경도를 입력합니다.")


@tool(args_schema=WeatherInput)
def get_weather(latitude, longitude) -> str:
    """
    제공된 좌표의 현재 기온을 섭씨(Celsius) 단위로 가져옵니다.
    """
    print("get_weather 도구 호출됨")
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}&current=temperature_2m",
            timeout=10,
        )
        data = response.json()
        temp = data["current"]["temperature_2m"]
        return f"현재 기온: {temp}°C"
    except Exception as e:
        return f"날씨 정보를 가져오는 중 오류 발생: {e}"


# ---------------------------------------------------------------------------------
# LLM 및 ReAct Agent 초기화 (서버 시작 시 1회만 생성해서 재사용)
# - 102_Memory_Concepts 노트북과 동일하게 InMemorySaver를 checkpointer로 사용
#   -> 같은 thread_id로 호출하면 이전 대화를 자동으로 기억 (단기 메모리)
#   -> 서버 프로세스가 종료되면 대화 이력도 사라짐
#     (영구 저장이 필요하면 SqliteSaver로 교체 가능)
# ---------------------------------------------------------------------------------
llm = init_chat_model(MODEL_NAME, model_provider="openai")
checkpointer = InMemorySaver()
agent = create_agent(
    llm,
    [search_db, get_weather],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# ---------------------------------------------------------------------------------
# 대화 스레드 관리
# - 대화 이력 자체는 checkpointer가 thread_id별로 저장하므로,
#   서버는 "현재 사용 중인 thread_id"만 관리하면 됨
# - 대화 초기화는 새 thread_id를 발급하는 방식으로 구현 (빈 대화에서 새로 시작)
# - threaded=True로 실행하면 요청이 동시에 처리될 수 있으므로 Lock으로 보호
# ---------------------------------------------------------------------------------
thread_counter = 1
thread_lock = threading.Lock()


def current_config():
    """현재 thread_id로 에이전트 호출용 config 생성"""
    with thread_lock:
        return {"configurable": {"thread_id": f"conversation_{thread_counter}"}}


# ---------------------------------------------------------------------------------
# Flask 앱과 라우트
# ---------------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    """채팅 UI 페이지 반환"""
    return send_file(BASE_DIR / "101_Tools_Memory_Agent.html")


@app.post("/chat")
def chat():
    """사용자 메시지를 받아 에이전트를 실행하고 답변을 JSON으로 반환

    이전 대화 이력은 checkpointer가 thread_id 기준으로 자동 관리하므로,
    매 요청마다 새 사용자 메시지 하나만 전달하면 됨
    """
    body = request.get_json(silent=True) or {}
    user_input = (body.get("message") or "").strip()
    if not user_input:
        return jsonify(error="메시지가 비어 있습니다."), 400

    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            current_config(),
        )
        ai_msg = response["messages"][-1]  # 마지막 메시지 추출
        reply = ai_msg.content or "응답이 생성되지 않았습니다."
    except Exception as e:
        reply = f"에러가 발생했습니다: {e}"

    return jsonify(reply=reply)


@app.post("/reset")
def reset():
    """대화 기록 초기화 - 새 thread_id를 발급하여 빈 대화에서 새로 시작"""
    global thread_counter
    with thread_lock:
        thread_counter += 1
    return jsonify(ok=True)


@app.post("/summary")
def summary():
    """checkpointer에 저장된 현재 스레드의 대화 이력을 LLM으로 요약하여 반환"""
    # 저장된 대화 이력 조회 (102 노트북의 '대화 이력 조회'에 해당)
    state = agent.get_state(current_config())
    history = state.values.get("messages", [])

    if not history:
        return jsonify(summary="요약할 대화가 없습니다.")

    conversation_text = []
    # 메시지 역할별로 텍스트로 변환 (도구 호출 등 그 외 메시지는 제외)
    for msg in history:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage) and msg.content:
            role = "AI"
        else:
            continue
        conversation_text.append(f"{role}: {msg.content}")
    joined_conversation = "\n".join(conversation_text)

    # 요약 프롬프트 생성 후 LLM에게 요약 요청
    prompt_content = f"""다음 대화를 요약해주세요:\n{joined_conversation}\n--- \n요약:\n"""
    try:
        summary_response = llm.invoke([HumanMessage(content=prompt_content)])
        summary_text = summary_response.content
    except Exception as e:
        summary_text = f"요약 중 에러가 발생했습니다: {e}"

    return jsonify(summary=summary_text)


if __name__ == "__main__":
    print(f"ReAct Agent 챗봇 서버 시작: http://localhost:{PORT}  (종료: Ctrl+C)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
