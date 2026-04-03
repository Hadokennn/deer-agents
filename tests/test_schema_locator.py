"""Tests for tools/schema_locator.py — pure helper functions + mock MCP integration."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.schema_locator import (
    SchemaLocatorTool,
    build_config_dimension,
    extract_category_levels,
    extract_field_summary,
    list_all_fields,
    parse_mcp_response,
    save_schema_to_file,
    search_field_in_schema,
)


# ---------------------------------------------------------------------------
# parse_mcp_response
# ---------------------------------------------------------------------------


def test_parse_mcp_response_list_format():
    """MCP tools return list[{"type":"text","text":"..."}]."""
    raw = [{"type": "text", "text": '[{"config_dimension":{"category_id":"100"}}]'}]
    result = parse_mcp_response(raw)
    assert isinstance(result, list)
    assert result[0]["config_dimension"]["category_id"] == "100"


def test_parse_mcp_response_string():
    result = parse_mcp_response('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_mcp_response_empty_list():
    result = parse_mcp_response([{"type": "text", "text": "[]"}])
    assert result == []


def test_parse_mcp_response_passthrough():
    """Non-string, non-list values pass through."""
    result = parse_mcp_response(42)
    assert result == 42


# ---------------------------------------------------------------------------
# build_config_dimension
# ---------------------------------------------------------------------------


def test_build_config_dimension_from_strings():
    """locate_template returns string values; build should convert to int."""
    dim = {
        "category_id": "5019003",
        "product_type": "1",
        "product_sub_type": "0",
        "template_type": "1",
        "template_sub_type": "0",
    }
    result = build_config_dimension(dim)
    assert result["category_id"] == 5019003
    assert result["product_type"] == 1
    assert isinstance(result["category_id"], int)


def test_build_config_dimension_defaults():
    """Missing optional fields get defaults."""
    dim = {"category_id": "100", "product_type": "11"}
    result = build_config_dimension(dim)
    assert result["product_sub_type"] == 0
    assert result["template_type"] == 1
    assert result["template_sub_type"] == 0


# ---------------------------------------------------------------------------
# extract_category_levels
# ---------------------------------------------------------------------------


def test_category_levels_three():
    levels = extract_category_levels("购物>果蔬生鲜>水果")
    assert levels == ["购物>果蔬生鲜>水果", "购物>果蔬生鲜", "购物"]


def test_category_levels_single():
    levels = extract_category_levels("购物")
    assert levels == ["购物"]


def test_category_levels_with_spaces():
    levels = extract_category_levels("购物 > 果蔬生鲜 > 水果")
    assert levels == ["购物>果蔬生鲜>水果", "购物>果蔬生鲜", "购物"]


# ---------------------------------------------------------------------------
# search_field_in_schema (nested group structure)
# ---------------------------------------------------------------------------

# Real-world structure: groups → fields
SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "basicInfo": {
            "type": "void",
            "x-component": "CommonGroupItem",
            "x-component-props": {"label": "基础信息"},
            "properties": {
                "categoryId": {
                    "title": "商品品类",
                    "type": "number",
                    "x-component": "CategorySelect",
                    "attr_keys_scope": {"bind_attr_keys": ["category_id"]},
                    "reaction_rules": [],
                },
            },
        },
        "merchantInfo": {
            "type": "void",
            "x-component": "CommonGroupItem",
            "x-component-props": {"label": "商家信息"},
            "properties": {
                "accountName": {
                    "title": "商家名称",
                    "type": "string",
                    "x-component": "AccountName",
                    "attr_keys_scope": {"bind_attr_keys": ["account_name"]},
                    "required": True,
                    "reaction_rules": [{"actionTypeEnum": "VISIBLE"}],
                },
                "platformProductId": {
                    "title": "商家平台商品ID",
                    "type": "string",
                    "x-component": "PlatformProductId",
                    "attr_keys_scope": {"bind_attr_keys": ["platform_product_id"]},
                    "reaction_rules": [{"actionTypeEnum": "VISIBLE"}],
                },
            },
        },
    },
}


def test_search_by_title_exact():
    result = search_field_in_schema(SAMPLE_SCHEMA, "商家名称")
    assert result is not None
    assert result["key"] == "accountName"
    assert result["group"] == "merchantInfo"


def test_search_by_field_key():
    result = search_field_in_schema(SAMPLE_SCHEMA, "accountName")
    assert result is not None
    assert result["key"] == "accountName"


def test_search_by_bind_attr_key():
    result = search_field_in_schema(SAMPLE_SCHEMA, "category_id")
    assert result is not None
    assert result["key"] == "categoryId"
    assert result["group"] == "basicInfo"


def test_search_by_component_name():
    result = search_field_in_schema(SAMPLE_SCHEMA, "PlatformProductId")
    assert result is not None
    assert result["key"] == "platformProductId"


def test_search_fuzzy_title():
    result = search_field_in_schema(SAMPLE_SCHEMA, "平台商品ID")
    assert result is not None
    assert result["key"] == "platformProductId"


def test_search_not_found():
    result = search_field_in_schema(SAMPLE_SCHEMA, "不存在的字段")
    assert result is None


def test_search_empty_schema():
    result = search_field_in_schema({}, "anything")
    assert result is None


# ---------------------------------------------------------------------------
# list_all_fields
# ---------------------------------------------------------------------------


def test_list_all_fields():
    fields = list_all_fields(SAMPLE_SCHEMA)
    assert len(fields) == 3
    titles = [f["title"] for f in fields]
    assert "商品品类" in titles
    assert "商家名称" in titles


# ---------------------------------------------------------------------------
# save_schema_to_file
# ---------------------------------------------------------------------------


def test_save_schema_creates_file(tmp_path):
    config_dim = {"category_id": 5019003, "product_type": 1}
    schema = {"properties": {"basicInfo": {"properties": {"f1": {"title": "test"}}}}}
    path = save_schema_to_file(config_dim, schema, schema_dir=tmp_path)
    assert path.exists()
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["properties"]["basicInfo"]["properties"]["f1"]["title"] == "test"


def test_save_schema_filename_is_safe(tmp_path):
    config_dim = {"category_id": 100, "product_type": 1, "template_type": 1}
    path = save_schema_to_file(config_dim, {}, schema_dir=tmp_path)
    assert "{" not in path.name
    assert '"' not in path.name


# ---------------------------------------------------------------------------
# extract_field_summary
# ---------------------------------------------------------------------------


def test_extract_field_summary():
    field = SAMPLE_SCHEMA["properties"]["merchantInfo"]["properties"]["accountName"]
    summary = extract_field_summary(field)
    assert summary["title"] == "商家名称"
    assert summary["x_component"] == "AccountName"
    assert summary["required"] is True
    assert len(summary["reaction_rules"]) == 1


def test_extract_field_summary_defaults():
    summary = extract_field_summary({"title": "minimal"})
    assert summary["required"] is False
    assert summary["x_disabled"] is False
    assert summary["reaction_rules"] == []
    assert summary["validator_rules"] == []


# ---------------------------------------------------------------------------
# SchemaLocatorTool with mock MCP
# ---------------------------------------------------------------------------


def _make_mock_mcp_tools(locate_result, detail_result):
    """Create mock MCP tools that return MCP-format responses."""
    locate_tool = MagicMock()
    # MCP returns list[{"type":"text","text":"..."}]
    locate_tool.invoke.return_value = [
        {"type": "text", "text": json.dumps(locate_result)}
    ]

    detail_tool = MagicMock()
    detail_tool.invoke.return_value = [
        {"type": "text", "text": json.dumps(detail_result)}
    ]

    return {
        "bytedance-mcp-ace_ai_ace_ai_locate_template": locate_tool,
        "bytedance-mcp-ace_ai_ace_ai_get_last_template_detail": detail_tool,
    }


def _wrap_locate_result(dim: dict) -> list:
    """Wrap a dimension dict in locate_template response format."""
    return [{"config_dimension": dim}]


SAMPLE_DIM = {
    "category_full_name": "购物>果蔬生鲜>水果",
    "category_id": "5019003",
    "product_type": "1",
    "product_sub_type": "0",
    "template_type": "1",
    "template_sub_type": "0",
}

SAMPLE_DETAIL = {
    "online_template": [
        {
            "template_name": "test",
            "attr_list": [],
            "schema_config": json.dumps(SAMPLE_SCHEMA),
        }
    ]
}


def test_tool_found(tmp_path):
    """Happy path: template found, field found."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
        field_name="商家名称",
    )

    assert result["status"] == "found"
    assert result["field_key"] == "accountName"
    assert result["group"] == "merchantInfo"
    assert result["x_component"] == "AccountName"
    assert "reaction_rules" in result
    assert Path(result["schema_path"]).exists()


def test_tool_ambiguous(tmp_path):
    """Multiple templates match — return candidates."""
    dims = [
        {"config_dimension": {**SAMPLE_DIM, "product_type": "1"}},
        {"config_dimension": {**SAMPLE_DIM, "product_type": "11"}},
    ]
    mcp_tools = _make_mock_mcp_tools(locate_result=dims, detail_result={})
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        field_name="any",
    )

    assert result["status"] == "ambiguous"
    assert len(result["candidates"]) == 2


def test_tool_template_not_found(tmp_path):
    """No template found at any category level."""
    mcp_tools = _make_mock_mcp_tools(locate_result=[], detail_result={})
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="不存在>的>类目",
        field_name="any",
    )

    assert result["status"] == "not_found"
    assert "suggestion" in result


def test_tool_field_not_found(tmp_path):
    """Template found but field not in schema."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
        field_name="不存在的字段",
    )

    assert result["status"] == "field_not_found"
    assert "available_fields" in result
    assert len(result["available_fields"]) > 0


def test_tool_overview_no_field_name(tmp_path):
    """No field_name → return overview with field list."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
    )

    assert result["status"] == "overview"
    assert result["field_count"] == 3
    assert len(result["fields"]) == 3
    assert Path(result["schema_path"]).exists()
    titles = [f["title"] for f in result["fields"]]
    assert "商品品类" in titles


def test_tool_found_with_component_sources(tmp_path, monkeypatch):
    """When field has x-component, component_sources are included."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))

    # Mock locate_component_code to return fake results
    import tools.schema_locator as sl
    monkeypatch.setattr(sl, "locate_component_code", lambda name: (
        "/repo/root",
        [{"name": name, "kind": "const/fn", "file": f"src/{name}.tsx", "line": 10, "span": 20, "exported": True}],
    ))

    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
        field_name="商家名称",
    )

    assert result["status"] == "found"
    assert result["x_component"] == "AccountName"
    assert "component_sources" in result
    assert len(result["component_sources"]) == 1
    assert result["component_sources"][0]["file"] == "src/AccountName.tsx"


def test_tool_found_no_component_when_index_empty(tmp_path, monkeypatch):
    """No component_sources when symbol index returns nothing."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))

    import tools.schema_locator as sl
    monkeypatch.setattr(sl, "locate_component_code", lambda name: ("", []))

    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
        field_name="商家名称",
    )

    assert result["status"] == "found"
    assert "component_sources" not in result


def test_tool_sandbox_path_with_thread_id(tmp_path, monkeypatch):
    """When ensure_config has thread_id, schema saves to sandbox outputs dir."""
    mcp_tools = _make_mock_mcp_tools(
        locate_result=_wrap_locate_result(SAMPLE_DIM),
        detail_result=SAMPLE_DETAIL,
    )
    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))

    # Mock ensure_config to return a config with thread_id
    from unittest.mock import patch
    mock_config = {"configurable": {"thread_id": "test-thread-123"}, "tags": [], "metadata": {}, "callbacks": None, "recursion_limit": 25}
    with patch("langchain_core.runnables.ensure_config", return_value=mock_config):
        result = tool._run(
            category_full_name="购物>果蔬生鲜>水果",
            product_type="1",
            field_name="商家名称",
        )

    assert result["status"] == "found"
    # schema_path should be a virtual sandbox path
    assert result["schema_path"].startswith("/mnt/user-data/outputs/schemas/")
    # Physical file should exist in thread outputs dir
    physical = Path(f".deer-flow/threads/test-thread-123/user-data/outputs/schemas/")
    assert physical.exists()
    schema_files = list(physical.glob("*.json"))
    assert len(schema_files) == 1

    # Cleanup
    import shutil
    shutil.rmtree(".deer-flow/threads/test-thread-123", ignore_errors=True)


def test_tool_category_fallback(tmp_path):
    """Leaf category fails, parent category succeeds."""
    parent_dim = {**SAMPLE_DIM, "category_full_name": "购物>果蔬生鲜", "category_id": "5019000"}

    locate_tool = MagicMock()
    # First call (leaf) returns empty, second call (parent) returns template
    locate_tool.invoke.side_effect = [
        [{"type": "text", "text": "[]"}],
        [{"type": "text", "text": json.dumps(_wrap_locate_result(parent_dim))}],
    ]

    detail_tool = MagicMock()
    detail_tool.invoke.return_value = [
        {"type": "text", "text": json.dumps(SAMPLE_DETAIL)}
    ]

    mcp_tools = {
        "bytedance-mcp-ace_ai_ace_ai_locate_template": locate_tool,
        "bytedance-mcp-ace_ai_ace_ai_get_last_template_detail": detail_tool,
    }

    tool = SchemaLocatorTool(mcp_tools=mcp_tools, schema_dir=str(tmp_path))
    result = tool._run(
        category_full_name="购物>果蔬生鲜>水果",
        product_type="1",
        field_name="商家名称",
    )

    assert result["status"] == "found"
    assert result["category"] == "购物>果蔬生鲜"
    # Verify fallback: leaf failed, parent succeeded → 2 locate calls
    assert locate_tool.invoke.call_count == 2
