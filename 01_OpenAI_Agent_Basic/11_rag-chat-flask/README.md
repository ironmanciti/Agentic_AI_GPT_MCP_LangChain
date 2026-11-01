# 14. RAG 챗봇 - Flask + HTML

Streamlit 없이 **Flask API + HTML 한 장(바닐라 JavaScript)** 으로
RAG 챗봇 웹 UI를 구현한 예제입니다.

쇼핑몰 고객 지원 에이전트: 웹 검색(`WebSearchTool`) + RAG(`FileSearchTool` + OpenAI Vector Store)

## 실행 방법

```bash
cd 01_OpenAI_Agent_Basic/11_rag-chat-flask
python app.py
```

브라우저에서 http://localhost:8000 접속

## 필요 조건

- `.env`에 `OPENAI_API_KEY` 설정
- `../data/faqs.json`, `../data/knowledgeBase.json` (Vector Store 업로드용, 최초 1회)
- 패키지: `flask`, `openai`, `openai-agents`, `requests`, `python-dotenv`

## 구조

| 파일 | 역할 |
|------|------|
| `app.py` | Flask 서버 + 에이전트 실행 (라우트 3개: `/`, `/chat`, `/reset`) |
| `index.html` | 채팅 UI (바닐라 JavaScript의 `fetch`로 서버 호출) |

## 동작 흐름

```
브라우저                          app.py (Flask)
--------                          --------------
GET /            ─────────────→  index.html 반환
사용자 입력
fetch POST /chat ─────────────→  대화 기록에 추가
  {"message": "..."}             Runner.run_sync(agent, 전체 기록)
                 ←─────────────  {"reply": "..."}
말풍선 표시
POST /reset      ─────────────→  대화 기록 초기화
```

