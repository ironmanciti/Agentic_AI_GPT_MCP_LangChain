"""
LangChain RAG Agent 기반 Chatbot (Flask + HTML)

- 에이전트 구성은 103_RAG_Agent 노트북 내용을 반영:
  - 나이키 10-K PDF를 인덱싱(로드 -> 분할 -> 임베딩 -> 벡터 스토어 저장)하고,
    벡터 스토어 검색을 도구(retrieve_context)로 감싼 에이전트 기반 RAG(Agentic RAG)
  - InMemorySaver(checkpointer) 기반 단기 메모리로 대화 이력을 자동 관리
- 실행: python 103_RAG_Chatbot.py  ->  브라우저에서 http://localhost:8000 접속

라우트 구성:
  GET  /         채팅 UI(103_RAG_Chatbot.html) 반환
  POST /chat     {"message": "..."} 를 받아 에이전트 실행 후 {"reply": "..."} 반환
  POST /reset    대화 기록 초기화 (새 thread_id 발급)
  POST /summary  지금까지의 대화 내용을 LLM으로 요약하여 반환
"""

import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL_NAME = os.environ["OPENAI_MODEL"]
PORT = 8000
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "example_data" / "nke-10k-2023_korean.pdf"
SYSTEM_PROMPT = (
    "당신은 나이키 연례 보고서(10-K)에서 관련 문맥(context)을 검색하는 도구에 접근할 수 있습니다. "
    "사용자의 질문에 답하기 위해 먼저 retrieve_context 도구를 사용하여 관련 정보를 검색한 후, "
    "검색된 정보를 바탕으로 정확하고 유용한 답변을 제공하세요. "
    "검색 결과에 없는 내용은 모른다고 답하세요."
)

# ---------------------------------------------------------------------------------
# 인덱싱: RAG Agent를 위한 벡터 스토어 구축 (103_RAG_Agent 노트북 1장과 동일)
# - 서버 시작 시 1회만 수행: PDF 로드 -> 분할 -> 임베딩 -> 벡터 스토어 저장
# - 실습용 인메모리 방식이므로 서버를 재시작하면 다시 인덱싱함
#   (재인덱싱을 피하려면 Chroma 등 영구 저장 벡터 스토어로 교체 가능)
# ---------------------------------------------------------------------------------
print("PDF 문서 인덱싱 중... (서버 시작 시 1회, 수십 초 소요)")

loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크 최대 1000자
    chunk_overlap=200,    # 청크 간 200자 중첩
    add_start_index=True  # 원본 문서 내 시작 위치 저장
)
all_splits = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=all_splits)

print(f"인덱싱 완료: {len(docs)}페이지 -> {len(all_splits)}개 청크 저장")


# ---------------------------------------------------------------------------------
# 검색 도구 정의 (103_RAG_Agent 노트북 2장과 동일)
# - content_and_artifact: LLM에게 전달할 문자열(content)과 원본 문서(artifact)를 함께 반환
# ---------------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """질문에 답하기 위해 관련 정보를 검색합니다.

    이 도구는 벡터 스토어에서 질문과 관련된 문서를 검색합니다.
    검색된 문서는 답변 생성에 사용됩니다.

    Args:
        query: 검색할 질문 또는 키워드
    """
    # 벡터 스토어에서 유사한 문서 검색
    retrieved_docs = vector_store.similarity_search(query, k=2)

    # 검색된 문서를 문자열로 직렬화
    serialized = "\n\n".join(
        f"출처: {doc.metadata.get('source', 'N/A')} (페이지 {doc.metadata.get('page', 'N/A')})\n내용: {doc.page_content}"
        for doc in retrieved_docs
    )

    # 문자열과 문서 객체를 함께 반환
    return serialized, retrieved_docs


# ---------------------------------------------------------------------------------
# LLM 및 RAG Agent 초기화 (서버 시작 시 1회만 생성해서 재사용)
# - InMemorySaver를 checkpointer로 사용
#   -> 같은 thread_id로 호출하면 이전 대화를 자동으로 기억 (단기 메모리)
#   -> 서버 프로세스가 종료되면 대화 이력도 사라짐
# ---------------------------------------------------------------------------------
llm = init_chat_model(
    MODEL_NAME,
    model_provider="openai",
    use_responses_api=True,        # Responses API 사용 (Chat Completions에서는 추론 모델의 도구 호출 불가)
    reasoning={"effort": "low"},   # 추론 강도 낮춤 → 실습 속도·비용 절감
)
checkpointer = InMemorySaver()
agent = create_agent(
    llm,
    [retrieve_context],
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
    return send_file(BASE_DIR / "103_RAG_Chatbot.html")


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
        reply = ai_msg.text or "응답이 생성되지 않았습니다."
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
    # 저장된 대화 이력 조회
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
        elif isinstance(msg, AIMessage) and msg.text:
            role = "AI"
        else:
            continue
        conversation_text.append(f"{role}: {msg.text}")
    joined_conversation = "\n".join(conversation_text)

    # 요약 프롬프트 생성 후 LLM에게 요약 요청
    prompt_content = f"""다음 대화를 요약해주세요:\n{joined_conversation}\n--- \n요약:\n"""
    try:
        summary_response = llm.invoke([HumanMessage(content=prompt_content)])
        summary_text = summary_response.text
    except Exception as e:
        summary_text = f"요약 중 에러가 발생했습니다: {e}"

    return jsonify(summary=summary_text)


if __name__ == "__main__":
    print(f"RAG Agent 챗봇 서버 시작: http://localhost:{PORT}  (종료: Ctrl+C)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
