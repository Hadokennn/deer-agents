"""Tests for pipelines/resolver.py — ${var} template substitution."""

from pipelines.resolver import VariableResolver


def test_resolve_input_field():
    r = VariableResolver({"input": {"query": "hello"}})
    assert r.resolve("${input.query}") == "hello"


def test_resolve_step_output():
    r = VariableResolver({
        "input": {},
        "search": {"items": [1, 2, 3], "count": 3},
    })
    assert r.resolve("${search.count}") == 3
    assert r.resolve("${search.items}") == [1, 2, 3]


def test_resolve_nested_path():
    r = VariableResolver({
        "input": {},
        "step1": {"data": {"user": {"name": "alice"}}},
    })
    assert r.resolve("${step1.data.user.name}") == "alice"


def test_resolve_raw_value_preserves_type():
    r = VariableResolver({
        "input": {"flag": True, "count": 42, "obj": {"a": 1}},
    })
    assert r.resolve("${input.flag}") is True
    assert r.resolve("${input.count}") == 42
    assert r.resolve("${input.obj}") == {"a": 1}


def test_resolve_string_with_embedded_var_returns_str():
    r = VariableResolver({"input": {"name": "alice", "n": 3}})
    assert r.resolve("hello ${input.name}!") == "hello alice!"
    assert r.resolve("count=${input.n}") == "count=3"


def test_resolve_fallback_takes_first_truthy():
    r = VariableResolver({
        "input": {},
        "a": {"value": None},
        "b": {"value": "fallback"},
    })
    assert r.resolve("${a.value | b.value}") == "fallback"


def test_resolve_fallback_takes_first_value_when_present():
    r = VariableResolver({
        "input": {},
        "a": {"value": "primary"},
        "b": {"value": "fallback"},
    })
    assert r.resolve("${a.value | b.value}") == "primary"


def test_resolve_fallback_to_literal_string():
    r = VariableResolver({
        "input": {},
        "a": {"value": None},
    })
    assert r.resolve('${a.value | "default"}') == "default"


def test_resolve_fallback_all_missing_returns_none():
    r = VariableResolver({"input": {}})
    assert r.resolve("${a.value | b.value}") is None


def test_resolve_dict_recursively():
    r = VariableResolver({"input": {"name": "alice", "age": 30}})
    result = r.resolve({
        "user": "${input.name}",
        "meta": {"age": "${input.age}"},
    })
    assert result == {"user": "alice", "meta": {"age": 30}}


def test_resolve_list_recursively():
    r = VariableResolver({"input": {"a": 1, "b": 2}})
    result = r.resolve(["${input.a}", "${input.b}", "literal"])
    assert result == [1, 2, "literal"]


def test_resolve_passthrough_non_string():
    r = VariableResolver({})
    assert r.resolve(42) == 42
    assert r.resolve(True) is True
    assert r.resolve(None) is None


def test_resolve_handles_none_step_result():
    r = VariableResolver({"input": {}, "skipped_step": None})
    assert r.resolve("${skipped_step.field | \"default\"}") == "default"


def test_resolve_missing_path_in_fallback_skipped():
    r = VariableResolver({"input": {}, "step1": {"a": 1}})
    assert r.resolve("${step1.b | step1.a}") == 1
