"""Tests for codeact/sandbox.py — restricted Python code execution."""

import sys

import pytest

from codeact.errors import RestrictedImportError
from codeact.sandbox import CodeExecutionSandbox, ExecutionResult


def test_execute_simple_expression_returns_value():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("result = 1 + 1", namespace={})

    assert result.success
    assert result.return_value == 2
    assert result.exception is None


def test_execute_captures_stdout():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("print('hello'); print('world')", namespace={})

    assert result.success
    assert "hello" in result.stdout
    assert "world" in result.stdout


def test_execute_no_result_variable_returns_none():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("x = 5", namespace={})

    assert result.success
    assert result.return_value is None


def test_execute_exception_captured():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("result = 1 / 0", namespace={})

    assert not result.success
    assert result.exception is not None
    assert "ZeroDivisionError" in result.exception
    assert result.traceback is not None
    assert "ZeroDivisionError" in result.traceback


def test_execute_with_namespace_callable():
    sandbox = CodeExecutionSandbox()

    def add(a, b):
        return a + b

    result = sandbox.execute("result = add(2, 3)", namespace={"add": add})
    assert result.success
    assert result.return_value == 5


def test_execute_allowed_import_works():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("import json\nresult = json.dumps({'k': 'v'})", namespace={})

    assert result.success
    assert result.return_value == '{"k": "v"}'


def test_execute_blocked_import_raises_in_result():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("import os\nresult = os.getcwd()", namespace={})

    assert not result.success
    assert result.exception is not None
    assert "not allowed" in result.exception.lower() or "RestrictedImport" in result.exception


def test_execute_open_builtin_unavailable():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("result = open('/etc/passwd').read()", namespace={})

    assert not result.success
    assert result.exception is not None
    # 'open' should not be in the restricted builtins → NameError
    assert "open" in result.exception.lower() or "NameError" in result.exception


def test_execute_eval_builtin_unavailable():
    sandbox = CodeExecutionSandbox()
    result = sandbox.execute("result = eval('1+1')", namespace={})

    assert not result.success
    assert result.exception is not None
    assert "eval" in result.exception.lower() or "NameError" in result.exception


@pytest.mark.skipif(sys.platform == "win32", reason="signal.SIGALRM is Unix-only")
def test_execute_timeout_aborts_long_running_code():
    sandbox = CodeExecutionSandbox(timeout_seconds=0.5)
    code = "while True:\n    pass"
    result = sandbox.execute(code, namespace={})

    assert not result.success
    assert result.exception is not None
    assert "timeout" in result.exception.lower() or "exceeded" in result.exception.lower()


def test_execute_truncates_huge_stdout():
    sandbox = CodeExecutionSandbox(max_stdout_bytes=100)
    code = "for i in range(10000):\n    print('x' * 100)"
    result = sandbox.execute(code, namespace={})

    assert len(result.stdout) <= 100


def test_execute_can_use_safe_builtins():
    sandbox = CodeExecutionSandbox()
    code = """
xs = [3, 1, 2]
result = {
    "sorted": sorted(xs),
    "len": len(xs),
    "max": max(xs),
    "sum": sum(xs),
}
"""
    result = sandbox.execute(code, namespace={})
    assert result.success
    assert result.return_value == {"sorted": [1, 2, 3], "len": 3, "max": 3, "sum": 6}
