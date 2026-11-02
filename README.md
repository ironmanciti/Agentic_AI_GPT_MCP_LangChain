# GPT와 MCP로 구현하는 자율형 AI Agent 구축

OpenAI Agents SDK와 LangChain/LangGraph를 활용하여 자율형 AI Agent를 단계적으로 구현하는 학습 프로젝트입니다.

## 전체 폴더 구조

```
Agentic_AI_GPT_MCP/
├── 01_OpenAI_Agent_Basic/             # Part I: OpenAI Agents SDK 기초
│   ├── 01_Agents_Overview.ipynb ~ 12_multi_agent_orchestration.ipynb
│   ├── 11_rag-chat-flask/             # RAG 채팅 앱 (Flask + HTML)
│   └── data/                          # FAQ·지식 베이스 데이터
├── 02_OpenAI_MCP_Agent/               # Part I: MCP 연동 (stdio/http)
│   ├── mcp_server.py                  # FastMCP 서버 (의류 + Chinook DB)
│   ├── mcp_client.py                  # Flask 클라이언트 (stdio 기본, --http 옵션)
│   ├── mcp_index.html                 # 채팅 UI
│   └── mcp.json                       # MCP 서버 자동 기동 설정
└── 03_LangChain_Agent/                # Part II/III: LangChain + LangGraph
    ├── 101_Tools_Agents.ipynb ~ 340_Skills_Pattern.ipynb
    ├── Flask_WebUI/                   # Flask 실습 앱
    ├── 105_DB_MCP_Agent/              # Chinook DB MCP 서버 + Flask 챗봇
    ├── 106_Notion_DB_MCP_Total/       # Notion + Chinook 다중 MCP 통합 (Flask)
    └── example_data/                  # RAG 실습용 문서
```

### 노트북 종류

각 장의 본문 노트북(`NN_이름.ipynb`)에는 다음 보조 노트북이 함께 있습니다.

| 접두어 | 용도 |
|------|------|
| `practice_` | 실습 문제 — `# TODO` 부분을 채우며 실행 |
| `practice_answer_` | 실습 문제 모범답안 |
| `template_` | 본문 노트북에서 코드 셀을 비운 강의용 따라 치기 노트북 |
| `Template_Creation.ipynb` | 본문 노트북에서 `template_` 노트북을 생성하는 도구 |

---

## Part I: OpenAI Agent SDK만으로 구현 (LangChain 없음)

### 1. 01_OpenAI_Agent_Basic — Agent 기초 + RAG

| 파일 | 내용 | 실습 |
|------|------|:---:|
| `01_Agents_Overview.ipynb` | Agent, 핸드오프 기본, 도구, 가드레일, 구조화 출력 | ✅ |
| `02_agent_runner.ipynb` | Agent 실행 (동기/비동기, Runner) | |
| `03_error_handling.ipynb` | 에러 핸들링 | |
| `04_Handoffs.ipynb` | 핸드오프 심화 — `handoff()`, `on_handoff`, `input_type`, `input_filter` | ✅ |
| `05_sessions.ipynb` | 세션 관리 | |
| `07_guardrails.ipynb` | 입력·출력 가드레일 | ✅ |
| `08_agent_visualization.ipynb` | 에이전트 구조 시각화 (**Google Colab에서 실행**) | |
| `10_Chatbot_RAG_Agent.ipynb` | RAG 에이전트 구현 | ✅ |
| `12_multi_agent_orchestration.ipynb` | 코드 기반 멀티에이전트 오케스트레이션 (순차·병렬·라우팅) | |

> `08_agent_visualization.ipynb`는 Graphviz 실행 파일이 필요하므로 **Google Colab**에서 실행합니다.
> 첫 셀(`apt-get install graphviz`, `pip install openai-agents[viz]`)을 먼저 실행하고, `OPENAI_MODEL` 환경 변수를 설정하세요.

**실습 앱: `11_rag-chat-flask/`**

OpenAI Agents SDK와 RAG 도구(`@function_tool`)를 사용하는 Flask 채팅봇입니다.
FAQ 데이터(`data/faqs.json`)를 임베딩하여 유사 문서를 검색하고, 그 컨텍스트로 답변을 생성합니다.

```bash
cd 01_OpenAI_Agent_Basic/11_rag-chat-flask
python app.py
# 브라우저에서 http://localhost:8000 접속
```

---

### 2. 02_OpenAI_MCP_Agent — MCP 연동 (stdio / streamable HTTP)

하나의 MCP 서버(의류 + Chinook DB)에 **stdio / streamable HTTP** 두 가지 방식으로 연결하는 실습입니다.

**구성 파일**

| 파일 | 역할 |
|------|------|
| `mcp_server.py` | FastMCP 서버 — 의류(get_price, add_item, list_items) + Chinook(execute_sql_query, list_tables, get_table_schema) |
| `mcp.json` | stdio 방식에서 클라이언트가 서버를 subprocess로 자동 기동하는 설정 |
| `mcp_client.py` | Flask 클라이언트 — 기본 stdio, `--http` 옵션 시 Streamable HTTP |
| `mcp_index.html` | 채팅 UI (제목에 현재 전송 방식 표시) |

**실행 방법** (브라우저에서 http://localhost:8001 접속)

```bash
# stdio 방식: 클라이언트가 mcp.json 설정으로 서버를 자동 기동
python mcp_client.py

# streamable HTTP 방식: 서버를 먼저 띄운 뒤 접속
python mcp_server.py --http     # 별도 터미널 (port 8000)
python mcp_client.py --http
```

**필요 환경**
- 동일 폴더에 `Chinook.db`
- `.env`: `OPENAI_API_KEY`, `OPENAI_MODEL`
- 패키지: `openai-agents`, `flask`, `fastmcp`, `python-dotenv`

---

## Part II/III: LangChain + LangGraph 에이전트

### 3. 03_LangChain_Agent — LangChain 과정

LangChain v1(`create_agent`)과 LangGraph를 활용한 AI 에이전트 개발 과정입니다.

#### 커리큘럼

| 파일 | 내용 | 실습 |
|------|------|:---:|
| `101_Tools_Agents.ipynb` | Tools & Agents 기초 | ✅ |
| `102_Memory_Concepts.ipynb` | 단기 메모리 개념 및 관리 | ✅ |
| `103_RAG_Agent.ipynb` | RAG 에이전트 (검색 증강 생성) | ✅ |
| `104_HITL_Agent.ipynb` | Human-in-the-Loop — SQL 에이전트로 배우기 | ✅ |
|------|------|:---:|
| `310_Subagents_Pattern.ipynb` | 하위 에이전트 위임 패턴 | |
| `320_Handoffs_Pattern.ipynb` | 에이전트 간 핸드오프 패턴 | |
| `330_Router_Pattern.ipynb` | 라우터 패턴 | |
| `340_Skills_Pattern.ipynb` | 스킬 기반 패턴 (Progressive Disclosure) | |

#### 실습 앱 (`Flask_WebUI/`)

| 파일 | 연결 학습 | 실행 |
|------|-----------|------|
| `101_Tools_Memory_Agent.py` | Tools + 메모리 에이전트 (Flask Web UI) | `python 101_Tools_Memory_Agent.py` |
| `103_RAG_Chatbot.py` | 문서 기반 RAG 챗봇 (Flask Web UI) | `python 103_RAG_Chatbot.py` |
| `practice_101_Tools_Memory_Agent.py` | 101 앱 실습 (`# TODO` 채우기) | `python practice_101_Tools_Memory_Agent.py` |
| `practice_103_RAG_Chatbot.py` | 103 앱 실습 (`# TODO` 채우기) | `python practice_103_RAG_Chatbot.py` |

```bash
cd 03_LangChain_Agent/Flask_WebUI
python 101_Tools_Memory_Agent.py   # 브라우저에서 http://localhost:8000 접속
```

> **모범답안을 옆에 두고 실습하기 (VS Code)**
> - 비교 보기 (추천): 탐색기에서 모범답안(`101_Tools_Memory_Agent.py`) 클릭 → `Ctrl`+클릭으로 실습 파일 선택 → 우클릭 **선택 항목 비교(Compare Selected)**. 왼쪽이 모범답안, 오른쪽이 실습 파일이며, 채울 부분이 색으로 표시되고 오른쪽에서 바로 편집할 수 있습니다.
> - 좌우 분할: 실습 파일을 연 상태에서 `Ctrl+\` → 한쪽 창에서 `Ctrl+P`로 모범답안 열기

---

### 4. 03_LangChain_Agent/105_DB_MCP_Agent — Chinook DB MCP Agent

Model Context Protocol(MCP)을 사용하여 Chinook 데이터베이스를 분석하는 대화형 AI 에이전트입니다.

**파일 구조**

```
105_DB_MCP_Agent/
├── agent_server.py      # Chinook DB MCP 서버 (FastMCP, stdio)
├── flask_app.py         # Flask 웹 챗봇 (langchain.mcp.MCPAdapter 기반)
├── practice_agent_server.py  # MCP 서버 실습 (`# TODO` 채우기)
├── practice_flask_app.py     # Flask 챗봇 실습 (practice_agent_server.py 에 연결)
├── index.html           # 채팅 UI
└── Chinook.db           # 샘플 데이터베이스
```

**MCP 도구**

| 도구 | 설명 |
|------|------|
| `execute_sql_query(query)` | SQL 쿼리 실행 및 결과 반환 |
| `get_table_schema(table_name)` | 테이블 스키마(컬럼·타입·제약조건) 조회 |
| `list_tables()` | 전체 테이블 목록 반환 |

**실행 방법**

```bash
# Flask 웹 챗봇 (MCP 서버 자동 기동)
python flask_app.py
# 브라우저에서 http://127.0.0.1:5000 접속

# 실습: practice_agent_server.py → practice_flask_app.py 순서로 TODO를 채운 뒤
python practice_flask_app.py
```

> 실습 파일은 모범답안과 화면을 나눠 보면서 채웁니다 (방법은 `Flask_WebUI/` 실습 안내 참고).
> 서버 실습 전에 Flask 실습부터 하려면 `practice_flask_app.py`의 `_SERVER_SCRIPT`를 `"agent_server.py"`로 바꾸세요.

**Chinook 데이터베이스** — 디지털 미디어 스토어 샘플 DB

| 주요 테이블 | 내용 |
|------------|------|
| Customer | 고객 정보 |
| Invoice / InvoiceLine | 주문 및 주문 상세 |
| Track / Album / Artist | 음악 트랙·앨범·아티스트 |
| Genre | 장르 정보 |
| Employee | 직원 정보 |

> **참고**: `106_Notion_DB_MCP_Total`에서 이 서버를 subprocess로 재사용합니다.
> `Chinook.db` 파일은 `105_DB_MCP_Agent/` 폴더 안에 두어야 합니다 (서버가 `__file__` 기준으로 동일 디렉터리에서 DB를 탐색).

---

### 5. 03_LangChain_Agent/106_Notion_DB_MCP_Total — Notion + Chinook 다중 MCP 통합 (Flask)

**파일**: `flask_app.py`, `index.html`, `practice_flask_app.py`(실습, `# TODO` 채우기)

`105_DB_MCP_Agent`에 이어지는 실습 과제입니다. 105의 Flask 앱 구조는 그대로 유지하면서,
`MCPAdapter` 서버 목록(`mcpServers`)에 공식 Notion MCP 서버(`@notionhq/notion-mcp-server`, npx 실행)
항목 하나만 추가하여 DB 조회 + Notion 기록이 모두 가능한 통합 Agent를 만듭니다.

- Chinook DB를 SQL로 분석하고, 결과를 Notion 페이지로 자동 생성
- Chinook MCP 서버는 `03_LangChain_Agent/105_DB_MCP_Agent/agent_server.py`를 재사용
- `NOTION_PAGE_ID`는 시스템 프롬프트로 주입되어 기본 작업 페이지로 사용
- 105 앱(포트 5000)과 동시 실행할 수 있도록 포트 5001 사용

> **설정 필수**: `Chinook.db`를 `03_LangChain_Agent/105_DB_MCP_Agent/` 폴더에 배치하세요.

**실행**:

```bash
python flask_app.py
# 브라우저에서 http://127.0.0.1:5001 접속

# 실습: TODO를 채운 뒤
python practice_flask_app.py
# 예: "매출 상위 5개 국가를 조회해서 Notion 페이지로 정리해줘"
```

---

## 공통 환경 설정

```bash
conda create -n langchain python=3.12 -y
conda activate langchain
pip install -r requirements.txt
```

`requirements.txt`에는 강의 코드가 실제로 사용하는 패키지와 그 의존성만 버전 고정으로 들어 있습니다 (Windows, Python 3.12 기준).

루트의 `sample.env`를 `.env`로 복사한 뒤 API 키를 채웁니다. 하위 폴더에서 실행해도 루트 `.env`를 찾아 사용합니다.

```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key        # LangSmith 트레이싱 시 (선택)
NOTION_API_KEY=your_notion_integration_token    # Notion MCP 사용 시
NOTION_PAGE_ID=your_default_notion_page_id      # Notion MCP 사용 시

OPENAI_MODEL=gpt-6-luna                         # 모든 노트북/앱의 기본 모델 (여기서 일괄 변경)
OPENAI_MODEL_LARGE=gpt-6.1-sol                  # 고성능 모델 예제용 (07_guardrails)
OPENAI_MODEL_SMALL=                             # 경량 모델 (라우터·분류 등 단순 작업용)
```

> LangChain 노트북·앱은 모두 `init_chat_model(os.environ["OPENAI_MODEL"], use_responses_api=True, reasoning={"effort": "low"})` 형식으로 모델을 생성합니다.

## 주의사항

- API 키는 반드시 `.env` 파일에 저장하고 버전 관리에 포함하지 마세요
- Chinook DB를 사용하는 실습(`02_OpenAI_MCP_Agent`, `104_HITL_Agent`, `105_DB_MCP_Agent`, `106_Notion_DB_MCP_Total`)은 `Chinook.db` 파일이 필요합니다
- SQL 에이전트 사용 시 보안을 고려하여 읽기 전용 권한을 권장합니다

## 참고 자료

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [LangChain 공식 문서](https://docs.langchain.com/oss/python/learn)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [Notion MCP Server](https://github.com/notionhq/notion-mcp-server)
