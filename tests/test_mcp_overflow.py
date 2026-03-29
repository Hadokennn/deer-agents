from pathlib import Path


def test_small_response_passes_through():
    """Responses under threshold are not modified."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    mw = ToolResponseProcessorMiddleware(max_response_size=100)
    result = mw.process_tool_response(content="short response", tool_call_id="call-1")
    assert result == "short response"


def test_exact_threshold_passes_through():
    """Response exactly at threshold is not replaced."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    mw = ToolResponseProcessorMiddleware(max_response_size=100)
    content = "x" * 100
    result = mw.process_tool_response(content=content, tool_call_id="call-1")
    assert result == content


def test_non_string_content_passes_through():
    """Non-string content (e.g., list) is not processed."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    mw = ToolResponseProcessorMiddleware(max_response_size=10)
    result = mw.process_tool_response(content=["not", "a", "string"], tool_call_id="call-1")
    assert result == ["not", "a", "string"]


def test_search_results_extracted(tmp_path):
    """web_search results are structurally extracted, not just truncated."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    # max_response_size: extracted summary must fit, raw content must exceed
    mw = ToolResponseProcessorMiddleware(max_response_size=600, sandbox_path=str(tmp_path) + "/")

    search_content = "\n".join([
        "Title: Python 3.14 Release Notes",
        "URL: https://docs.python.org/3/whatsnew/3.14.html",
        "Content: Python 3.14 introduces t-strings.",
        "",
        "Title: Best New Features",
        "URL: https://example.com",
        "Content: Template strings and JIT.",
        "",
    ] * 10)  # ~2KB raw, ~500B extracted (top 5 results)

    result = mw.process_tool_response(
        content=search_content, tool_call_id="call-search", tool_name="web_search"
    )

    # Should contain extracted titles, not raw content
    assert "Python 3.14 Release Notes" in result
    assert "Processed" in result  # Processing indicator
    assert len(result) < len(search_content)

    # Full content saved to sandbox
    saved = tmp_path / "call-search.txt"
    assert saved.exists()
    assert saved.read_text() == search_content


def test_log_content_extracted(tmp_path):
    """Log content extracts ERROR/WARN lines with context."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    mw = ToolResponseProcessorMiddleware(max_response_size=500, sandbox_path=str(tmp_path) + "/")

    log_lines = []
    for i in range(100):
        if i == 42:
            log_lines.append("2026-03-29 10:00:42 ERROR redis.ConnectionError: Connection timed out")
        elif i == 43:
            log_lines.append("2026-03-29 10:00:43 ERROR retry failed after 3 attempts")
        else:
            log_lines.append(f"2026-03-29 10:00:{i:02d} INFO normal operation line {i}")
    log_content = "\n".join(log_lines)

    result = mw.process_tool_response(
        content=log_content, tool_call_id="call-logs", tool_name="log_search"
    )

    # Should extract error lines with stats
    assert "errors" in result
    assert "redis.ConnectionError" in result
    assert len(result) < len(log_content)

    # Full saved to sandbox
    assert (tmp_path / "call-logs.txt").exists()


def test_large_unknown_tool_gets_truncated(tmp_path):
    """Unknown large tool responses get smart truncation with head + tail."""
    from middlewares.mcp_overflow import ToolResponseProcessorMiddleware

    mw = ToolResponseProcessorMiddleware(max_response_size=100, sandbox_path=str(tmp_path) + "/")

    content = "HEAD_DATA " * 500 + "TAIL_DATA " * 500
    result = mw.process_tool_response(content=content, tool_call_id="call-big", tool_name="unknown_tool")

    assert "HEAD_DATA" in result
    assert "TAIL_DATA" in result
    assert "truncated" in result
    assert len(result) < len(content)

    # Full saved
    assert (tmp_path / "call-big.txt").exists()


def test_backward_compat_alias():
    """McpOverflowMiddleware alias still works."""
    from middlewares.mcp_overflow import McpOverflowMiddleware, ToolResponseProcessorMiddleware
    assert McpOverflowMiddleware is ToolResponseProcessorMiddleware
