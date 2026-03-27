from pathlib import Path


def test_small_response_passes_through():
    """Responses under threshold are not modified."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    mw = McpOverflowMiddleware(max_response_size=100)
    result = mw.process_tool_response(content="short response", tool_call_id="call-1")
    assert result == "short response"


def test_large_response_gets_replaced(tmp_path):
    """Responses over threshold are written to file and replaced with pointer."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    sandbox_path = str(tmp_path) + "/"
    mw = McpOverflowMiddleware(max_response_size=50, sandbox_path=sandbox_path)

    big_content = "x" * 2000
    result = mw.process_tool_response(content=big_content, tool_call_id="call-abc")

    # Result should be a pointer, not the original content
    assert "call-abc" in result
    assert "read_file" in result
    assert len(result) < len(big_content)

    # File should exist on disk
    written_file = Path(sandbox_path) / "call-abc.txt"
    assert written_file.exists()
    assert written_file.read_text() == big_content


def test_exact_threshold_passes_through():
    """Response exactly at threshold is not replaced."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    mw = McpOverflowMiddleware(max_response_size=100)
    content = "x" * 100
    result = mw.process_tool_response(content=content, tool_call_id="call-1")
    assert result == content


def test_non_string_content_passes_through():
    """Non-string content (e.g., list) is not processed."""
    from middlewares.mcp_overflow import McpOverflowMiddleware

    mw = McpOverflowMiddleware(max_response_size=10)
    result = mw.process_tool_response(content=["not", "a", "string"], tool_call_id="call-1")
    assert result == ["not", "a", "string"]
