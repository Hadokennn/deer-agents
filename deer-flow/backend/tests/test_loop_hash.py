"""Tests for loop_hash utility module."""

from deerflow.agents.middlewares.loop_hash import (
    hash_tool_calls,
    normalize_tool_call_args,
    stable_tool_key,
)


def _bash_call(cmd="ls"):
    return {"name": "bash", "id": f"call_{cmd}", "args": {"command": cmd}}


class TestHashToolCalls:
    def test_same_calls_same_hash(self):
        assert hash_tool_calls([_bash_call("ls")]) == hash_tool_calls([_bash_call("ls")])

    def test_different_calls_different_hash(self):
        assert hash_tool_calls([_bash_call("ls")]) != hash_tool_calls([_bash_call("pwd")])

    def test_order_independent(self):
        a = hash_tool_calls([_bash_call("ls"), {"name": "read_file", "args": {"path": "/tmp"}}])
        b = hash_tool_calls([{"name": "read_file", "args": {"path": "/tmp"}}, _bash_call("ls")])
        assert a == b


class TestNormalizeArgs:
    def test_dict_passthrough(self):
        result, fallback = normalize_tool_call_args({"a": 1})
        assert result == {"a": 1}
        assert fallback is None

    def test_string_json_parsed(self):
        result, fallback = normalize_tool_call_args('{"a": 1}')
        assert result == {"a": 1}
        assert fallback is None

    def test_invalid_json_string_to_fallback(self):
        result, fallback = normalize_tool_call_args("not-json")
        assert result == {}
        assert fallback == "not-json"


class TestStableToolKey:
    def test_read_file_buckets_lines(self):
        a = stable_tool_key("read_file", {"path": "/x.py", "start_line": 1, "end_line": 10}, None)
        b = stable_tool_key("read_file", {"path": "/x.py", "start_line": 5, "end_line": 50}, None)
        assert a == b   # same 200-line bucket

    def test_write_file_uses_full_args(self):
        a = stable_tool_key("write_file", {"path": "/x.py", "content": "v1"}, None)
        b = stable_tool_key("write_file", {"path": "/x.py", "content": "v2"}, None)
        assert a != b
