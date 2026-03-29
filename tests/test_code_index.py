"""Tests for CodeIndexMiddleware — symbol search extraction and enrichment."""


def test_extract_grep_command():
    """Extract search query from bash grep targeting a code repo."""
    from middlewares.code_index import CodeIndexMiddleware

    mw = CodeIndexMiddleware(code_repos=[
        {"name": "my-repo", "path": "/code/my-repo"},
    ])

    # grep with quotes
    result = mw._extract_search_query({
        "name": "bash",
        "args": {"command": 'grep -r "ConnectionError" /code/my-repo/src/'},
    })
    assert result is not None
    assert result[0] == "ConnectionError"
    assert result[1] == "my-repo"


def test_extract_rg_command():
    """Extract from ripgrep command."""
    from middlewares.code_index import CodeIndexMiddleware

    mw = CodeIndexMiddleware(code_repos=[
        {"name": "my-repo", "path": "/code/my-repo"},
    ])

    result = mw._extract_search_query({
        "name": "bash",
        "args": {"command": "rg -n 'usePresale' /code/my-repo"},
    })
    assert result is not None
    assert result[0] == "usePresale"


def test_no_match_for_unrelated_bash():
    """Non-code-search bash commands return None."""
    from middlewares.code_index import CodeIndexMiddleware

    mw = CodeIndexMiddleware(code_repos=[
        {"name": "my-repo", "path": "/code/my-repo"},
    ])

    result = mw._extract_search_query({
        "name": "bash",
        "args": {"command": "echo hello"},
    })
    assert result is None


def test_format_index_results():
    """Index results formatted as concise context."""
    from middlewares.code_index import CodeIndexMiddleware

    mw = CodeIndexMiddleware()
    results = [
        {"name": "RedisClient", "kind": "class", "file": "src/cache.ts", "line": 10, "span": 50, "exported": True},
        {"name": "getRedis", "kind": "const/fn", "file": "src/cache.ts", "line": 62, "span": 8, "exported": False},
    ]

    output = mw._format_index_results(results, "redis")
    assert "Symbol index" in output
    assert "RedisClient" in output
    assert "src/cache.ts:10" in output
    assert "export" in output
