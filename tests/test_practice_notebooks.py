"""실습 노트북(practice_*)과 모범답안(practice_answer_*) 쌍의 형식 검증.

- 모든 모범답안에 대응하는 실습 노트북이 있다
- 실습 노트북: 제목/안내문 형식, 실행 결과 없음, TODO 존재, 코드 셀 문법 오류 없음
- 답안과 달라진 코드 셀에는 반드시 # TODO 가 있다 (준비 셀·markdown 셀은 그대로)
- 모범답안에는 TODO 빈칸이 남아 있지 않다
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("01_OpenAI_Agent_Basic", "03_LangChain_Agent")
TODO = "# TODO"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def src(cell: dict) -> str:
    return "".join(cell["source"])


def compiles(code: str) -> bool:
    """노트북 셀 코드가 문법적으로 올바른지 (최상위 await 허용)"""
    try:
        compile(code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except SyntaxError:
        return False
    return True


def practice_problems(answer: dict, practice: dict, answer_name: str) -> list[str]:
    """실습 노트북의 형식 문제 목록 (빈 리스트면 정상)"""
    problems = []
    a_cells, p_cells = answer["cells"], practice["cells"]
    if len(a_cells) != len(p_cells):
        return [f"셀 개수 다름: 답안 {len(a_cells)} / 실습 {len(p_cells)}"]

    head = src(p_cells[0])
    if not head.startswith("# 실습 문제 - "):
        problems.append("제목이 '# 실습 문제 - '로 시작하지 않음")
    if f"(모범답안: `{answer_name}`)" not in head:
        problems.append("안내문에 모범답안 파일명이 없음")

    todo_count = 0
    for i, (a, p) in enumerate(zip(a_cells, p_cells)):
        if a["cell_type"] != p["cell_type"]:
            problems.append(f"[{i}] 셀 종류 다름")
            continue
        if p["cell_type"] != "code":
            if i > 0 and src(a) != src(p):
                problems.append(f"[{i}] markdown 셀이 답안과 다름")
            continue
        if p.get("outputs") or p.get("execution_count") is not None:
            problems.append(f"[{i}] 실행 결과가 남아 있음")
        if not compiles(src(p)):
            problems.append(f"[{i}] 문법 오류")
        todo_count += src(p).count(TODO)
        if src(a) != src(p) and TODO not in src(p):
            problems.append(f"[{i}] 답안과 다르지만 TODO 표시 없음")
    if todo_count == 0:
        problems.append("TODO 가 하나도 없음")
    return problems


def _pairs():
    for folder in FOLDERS:
        for answer in sorted((ROOT / folder).glob("practice_answer_*.ipynb")):
            yield answer, answer.with_name(answer.name.replace("practice_answer_", "practice_", 1))


PAIRS = list(_pairs())


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------
def test_pairs_found():
    assert len(PAIRS) == 13, [a.name for a, _ in PAIRS]  # 01: 6개, 03: 7개


@pytest.mark.parametrize(("answer", "practice"), PAIRS, ids=[p.name for _, p in PAIRS])
def test_practice_notebook_format(answer, practice):
    assert practice.exists(), f"{practice.name} 없음"
    assert practice_problems(load(answer), load(practice), answer.name) == []


@pytest.mark.parametrize(("answer", "practice"), PAIRS, ids=[a.name for a, _ in PAIRS])
def test_answer_notebook_has_no_blanks(answer, practice):
    nb = load(answer)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert TODO not in src(cell), f"{answer.name}[{i}]: 모범답안에 TODO 남음"
            assert compiles(src(cell)), f"{answer.name}[{i}]: 문법 오류"


# ---------------------------------------------------------------------------
# 경계값
# ---------------------------------------------------------------------------
def test_compiles_allows_top_level_await_and_ellipsis():
    assert compiles("result = await Runner.run(agent, 'q')")
    assert compiles("if ...:\n    pass\nx = ...")
    assert compiles("")  # 빈 코드 셀


# ---------------------------------------------------------------------------
# 실패 케이스: 검사 함수가 잘못된 실습 노트북을 잡아내는지
# ---------------------------------------------------------------------------
def _nb(head, code, outputs=None, count=None):
    return {"cells": [
        {"cell_type": "markdown", "source": [head]},
        {"cell_type": "code", "source": [code], "outputs": outputs or [], "execution_count": count},
    ]}


ANSWER = _nb("# 실습 문제 모범답안 - X", "x = 1")
GOOD_HEAD = "# 실습 문제 - X\n\n`# TODO` ... (모범답안: `a.ipynb`)"


@pytest.mark.parametrize(
    ("practice", "expected"),
    [
        (_nb(GOOD_HEAD, "# TODO: x 값\nx = ..."), []),
        (_nb("# 실습 문제 모범답안 - X\n(모범답안: `a.ipynb`)", "# TODO\nx = ..."), ["제목"]),
        (_nb(GOOD_HEAD, "# TODO\nx = ...", outputs=[{"text": "1"}], count=1), ["실행 결과"]),
        (_nb(GOOD_HEAD, "x = ..."), ["TODO 표시 없음", "TODO 가 하나도 없음"]),
        (_nb(GOOD_HEAD, "# TODO\nx = (..."), ["문법 오류"]),
    ],
    ids=["정상", "제목 미변경", "출력 남음", "TODO 누락", "문법 오류"],
)
def test_practice_problems_detects(practice, expected):
    problems = practice_problems(ANSWER, practice, "a.ipynb")
    assert len(problems) == len(expected), problems
    for word, problem in zip(expected, problems):
        assert word in problem
