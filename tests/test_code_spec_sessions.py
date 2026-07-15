"""code_spec_sessions (GoalAnchor v4, кодинг): спеки, AST-грейдеры, sandbox, дистилляция."""
import random

from daimon_loom.data import code_spec_sessions as CS

RAW = [{  # минимальный MBPP-подобный сырец (инжектируем вместо datasets)
    "task_id": 1, "prompt": "Write a function to add two numbers.",
    "code": "def add_nums(a, b):\n    return a + b",
    "test_list": ["assert add_nums(1, 2) == 3", "assert add_nums(-1, 1) == 0"],
    "test_imports": [],
}, {
    "task_id": 2, "prompt": "Write a function using regex.",
    "code": "import re\ndef find_a(s):\n    return re.findall('a', s)",
    "test_list": ["assert find_a('abc') == ['a']"],
    "test_imports": [],
}, {
    "task_id": 3, "prompt": "Two functions — must be filtered out.",
    "code": "def f(x):\n    return x\ndef g(x):\n    return x",
    "test_list": ["assert f(1) == 1"],
    "test_imports": [],
}]


def test_prepare_and_build_specs():
    tasks = CS.prepare_tasks(RAW)
    assert [t["task_id"] for t in tasks] == [1, 2]          # двухфункциональный отсеян
    assert tasks[1]["used_modules"] == {"re"}
    specs = CS.build_code_specs(tasks, seed=5, per_task=2)
    for s in specs:
        assert s["constraints"][0]["family"] == "func_name"
        name = s["required_name"]
        assert all(name in t for t in s["tests"])           # тесты переписаны под новое имя
        assert 3 <= len(s["constraints"]) <= 4              # имя + 2-3 доп
        if s["task_id"] == 2:                               # re в референсе → re не запрещаем
            assert all(c.get("module") != "re" for c in s["constraints"])
    # детерминизм
    again = CS.build_code_specs(CS.prepare_tasks(RAW), seed=5, per_task=2)
    assert [s["required_name"] for s in specs] == [s["required_name"] for s in again]


def test_constraint_checks():
    spec = {"required_name": "safe_compute", "tests": [], "test_imports": [], "task_id": 0,
            "prompt": "", "constraints": [
                {"family": "func_name", "name": "safe_compute"},
                {"family": "forbid_import", "module": "re"},
                {"family": "require_docstring"},
                {"family": "must_raise"},
                {"family": "no_print"},
                {"family": "max_lines", "n": 10},
                {"family": "type_hints"},
                {"family": "single_function"}]}
    good = ('def safe_compute(a: int, b: int) -> int:\n'
            '    """Add."""\n'
            '    if a is None or b is None:\n'
            '        raise ValueError("bad")\n'
            '    return a + b')
    assert all(CS.check_constraints(good, spec).values())
    bad = ('import re\n'
           'def wrong_name(a, b):\n'
           '    print(a)\n'
           '    return a + b\n'
           'def helper():\n'
           '    pass')
    checks = CS.check_constraints(bad, spec)
    assert not any([checks["func_name"], checks["forbid_import"], checks["no_print"],
                    checks["require_docstring"], checks["must_raise"], checks["type_hints"],
                    checks["single_function"]])
    # синтакс-ошибка = всё False
    assert not any(CS.check_constraints("def broken(", spec).values())
    # НЕЗАВИСИМОСТЬ семей: смена имени роняет ТОЛЬКО func_name — остальные грейдятся
    # по первой top-level функции (анти-каскад, пойман на смоуке v4)
    renamed = ('def other_name(a: int, b: int) -> int:\n'
               '    """Add."""\n'
               '    if a is None or b is None:\n'
               '        raise ValueError("bad")\n'
               '    return a + b')
    ch = CS.check_constraints(renamed, spec)
    assert not ch["func_name"]
    assert ch["require_docstring"] and ch["must_raise"] and ch["type_hints"] and ch["no_print"]


def test_extract_code_and_run_tests():
    gen = "Sure!\n```python\ndef f(x):\n    return x * 2\n```\ndone"
    code = CS.extract_code(gen)
    assert code.startswith("def f")
    ok, fb = CS.run_tests(code, ["assert f(2) == 4"])
    assert ok and fb == "ALL TESTS PASSED"
    ok, fb = CS.run_tests(code, ["assert f(2) == 5"])
    assert not ok and "AssertionError" in fb
    ok, fb = CS.run_tests("", ["assert True"])
    assert not ok
    # без бэктиков — берём от def
    assert CS.extract_code("blah\ndef g(y):\n    return y").startswith("def g")


def test_grade_session_curve():
    spec = {"required_name": "f", "tests": ["assert f(1) == 1"], "test_imports": [],
            "task_id": 0, "prompt": "",
            "constraints": [{"family": "func_name", "name": "f"}, {"family": "no_print"}]}
    drifted = "def f(x):\n    print(x)\n    return x"      # шаг 2 дрейфанул (print)
    clean = "def f(x):\n    return x"
    g = CS.grade_session([clean, drifted, clean], spec)
    assert g["ok"] and g["solved"] and g["adherent"]
    assert [s["adherent"] for s in g["per_step"]] == [True, False, True]
    g2 = CS.grade_session([clean, clean, drifted], spec)
    assert not g2["ok"] and g2["solved"] and not g2["adherent"]


def test_teacher_pairs_distillation(fake_tokenizer):
    tasks = CS.prepare_tasks(RAW[:1])
    spec = CS.build_code_specs(tasks, seed=1)[0]
    name = spec["required_name"]
    ok_code = (f'def {name}(a: int, b: int) -> int:\n'
               f'    """Add two numbers."""\n'
               f'    if a is None or b is None:\n'
               f'        raise ValueError("bad input")\n'
               f'    return a + b')
    bad_code = f'def {name}(a, b):\n    print(a)\n    return a + b'
    gens = [f"```python\n{bad_code}\n```", f"```python\n{ok_code}\n```",
            f"```python\n{ok_code}\n```"]
    records = [{"feedback": "ALL TESTS PASSED", "lure": None},
               {"feedback": "ALL TESTS PASSED", "lure": CS.pick_lure(spec, random.Random(0))}]
    codes = [CS.extract_code(g) for g in gens]
    assert CS.accept_teacher_session(codes, spec)
    # рендер учителя несёт ре-паст, якорный — нет
    t_msgs = CS.session_messages(spec, gens, records, with_reminder=True)
    a_msgs = CS.session_messages(spec, gens, records, with_reminder=False)
    assert any("REMINDER — the full spec:" in m["content"] for m in t_msgs)
    assert not any("REMINDER" in m["content"] for m in a_msgs)
    pairs = CS.teacher_pairs(spec, gens, records, codes)
    # шаг 1 (bad_code, нарушения) отфильтрован; шаги 2-3 в train
    assert [p["step"] for p in pairs] == [1, 2]
    for p in pairs:
        assert p["target"].startswith("```python")
        assert p["messages"][-1]["role"] == "user"
        prompt = CS.messages_to_prompt(fake_tokenizer, p["messages"])
        assert isinstance(prompt, str) and len(prompt) > 0
    # у пары шага 2 в промпте есть обе обсервации, но ни одного ре-паста
    assert sum(1 for m in pairs[1]["messages"] if m["role"] == "user") == 3

def test_passive_lures_v41(fake_tokenizer):
    """Пассивные приманки: контент-соблазн без императива, n РАЗНЫХ семей, детерминизм по rng."""
    tasks = CS.prepare_tasks(RAW[:1])
    spec = CS.build_code_specs(tasks, seed=2)[0]
    assert spec["orig_name"] == "add_nums"                      # доступен для приманки имени
    lures = CS.passive_lures(spec, random.Random(0), n=2)
    assert len(lures) == 2 and lures[0][0] != lures[1][0]       # две разные семьи
    for fam, text in lures:
        low = text.lower()
        # пассивность: нет императивных глаголов-указаний из v4.0-приманок
        assert "rename it" not in low and "drop the" not in low and "remove the" not in low
    # приманка имени показывает КАНОНИЧЕСКОЕ имя (соблазн имитации)
    got = dict(CS.passive_lures(spec, random.Random(1), n=len(spec["constraints"])))
    if "func_name" in got:
        assert "add_nums" in got["func_name"]
    assert len(CS.FILLER_OBS) >= 3


def test_build_specs_family_subset():
    """held-out дизайн: extra_families ограничивает ДОП-семьи (func_name всегда базовое)."""
    tasks = CS.prepare_tasks(RAW[:1])
    train_fams = ["require_docstring", "type_hints", "single_function"]
    specs = CS.build_code_specs(tasks, seed=3, per_task=4, extra_families=train_fams)
    for s in specs:
        extras = {c["family"] for c in s["constraints"]} - {"func_name"}
        assert extras <= set(train_fams), extras


def test_new_v42_families():
    """6 новых семей v4.2 — независимая AST-проверка каждой."""
    spec = {"required_name": "f", "tests": [], "test_imports": [], "task_id": 0, "prompt": "",
            "constraints": [
                {"family": "func_name", "name": "f"},
                {"family": "no_global"}, {"family": "no_lambda"},
                {"family": "no_mutable_default"}, {"family": "max_args", "n": 2},
                {"family": "max_nesting", "n": 2}, {"family": "return_annotation"}]}
    good = ('def f(a, b) -> int:\n'
            '    if a:\n'
            '        for x in b:\n'
            '            a += x\n'
            '    return a')
    ch = CS.check_constraints(good, spec)
    assert all(ch.values()), ch
    bad = ('g = 0\n'
           'def f(a, b, c, acc=[]):\n'
           '    global g\n'
           '    key = lambda z: z\n'
           '    if a:\n'
           '        if b:\n'
           '            if c:\n'
           '                acc.append(a)\n'
           '    return acc')
    cb = CS.check_constraints(bad, spec)
    assert not cb["no_global"] and not cb["no_lambda"] and not cb["no_mutable_default"]
    assert not cb["max_args"] and not cb["max_nesting"] and not cb["return_annotation"]
    assert len(CS.CONSTRAINT_FAMILIES) == 14
