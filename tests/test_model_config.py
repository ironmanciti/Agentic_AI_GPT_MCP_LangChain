"""모델명이 .env 한 곳에서만 관리되는지 검증하는 테스트."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".ipynb_checkpoints", "__pycache__", "tests", ".git", ".venv"}
# 하드코딩된 채팅 모델 문자열 ("gpt-5.4-mini", "openai:gpt-5-mini", "o4-mini" 등)
HARDCODED_MODEL = re.compile(r"""["'](?:openai:)?(?:gpt-\d[\w.\-]*|o\d(?:-mini)?)["']""")
MODEL_VARS = ("OPENAI_MODEL", "OPENAI_MODEL_LARGE")


def _source_files():
    for path in ROOT.rglob("*"):
        if SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.suffix in (".py", ".ipynb"):
            yield path


def _code_cells(path):
    """노트북이면 코드 셀 목록, .py 파일이면 파일 전체를 하나의 셀로 반환한다."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [text]
    nb = json.loads(text)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _files_using_env_model():
    return [p for p in _source_files() if any("OPENAI_MODEL" in c for c in _code_cells(p))]


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------
def test_no_hardcoded_model_in_sources():
    offenders = [
        f"{p.relative_to(ROOT)}: {m.group(0)}"
        for p in _source_files()
        for cell in _code_cells(p)
        for m in HARDCODED_MODEL.finditer(cell)
    ]
    assert offenders == [], "하드코딩된 모델명:\n" + "\n".join(offenders)


@pytest.mark.parametrize("env_name", ["sample.env", ".env"])
def test_env_file_defines_models(env_name):
    env_path = ROOT / env_name
    if not env_path.exists():
        pytest.skip(f"{env_name} 없음")
    values = dotenv_values(env_path)
    for var in MODEL_VARS:
        assert values.get(var), f"{env_name}에 {var} 값이 없습니다"


def test_env_model_loaded_before_use():
    files = _files_using_env_model()
    assert len(files) >= 40, f"OPENAI_MODEL 사용 파일 수가 너무 적음: {len(files)}"
    for path in files:
        seen_dotenv = seen_os = False
        for cell in _code_cells(path):
            if "OPENAI_MODEL" in cell:
                use_pos = cell.index("OPENAI_MODEL")
                head = cell[:use_pos]
                assert seen_dotenv or "load_dotenv(" in head, f"{path}: load_dotenv 누락"
                assert seen_os or re.search(r"^\s*import os\b", head, re.MULTILINE), f"{path}: import os 누락"
            seen_dotenv |= "load_dotenv(" in cell
            seen_os |= bool(re.search(r"^\s*import os\b", cell, re.MULTILINE))


# ---------------------------------------------------------------------------
# 경계값: 하위 폴더에서 실행해도 루트 .env를 찾아야 한다 (노트북 실행 환경과 동일)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "subdir",
    ["01_OpenAI_Agent_Basic", "03_LangChain_Agent/105_DB_MCP_Agent"],
)
def test_root_env_found_from_subfolder(subdir):
    if not (ROOT / ".env").exists():
        pytest.skip(".env 없음")
    env = {k: v for k, v in os.environ.items() if k not in MODEL_VARS}
    code = "from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ['OPENAI_MODEL'])"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=ROOT / subdir, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == dotenv_values(ROOT / ".env")["OPENAI_MODEL"]


# ---------------------------------------------------------------------------
# 실패 케이스
# ---------------------------------------------------------------------------
def test_missing_model_var_fails_loudly(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in MODEL_VARS}
    code = "from dotenv import load_dotenv; import os; load_dotenv(); os.environ['OPENAI_MODEL']"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "KeyError" in result.stderr


def test_detector_catches_hardcoded_model():
    samples = ['Model = "gpt-5.4-mini"', "model='openai:gpt-5-mini'", 'm = "o4-mini"']
    for s in samples:
        assert HARDCODED_MODEL.search(s), s
    assert not HARDCODED_MODEL.search('OpenAIEmbeddings(model="text-embedding-3-small")')
