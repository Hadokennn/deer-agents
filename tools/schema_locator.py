"""Schema Locator Tool — Programmatic tool for oncall field diagnosis.

Chains MCP calls to locate template schema and extract field schema fragments,
then locates the component source code via symbol index.
LLM only participates at entry (intent) and exit (analysis). All intermediate
MCP calls, index lookups, and data processing are code-driven — no LLM in the loop.

Flow:
  1. MCP: locate_template (category_full_name + product_type → 五元组)
  2. Category tree fallback (leaf → parent → grandparent)
  3. MCP: get_last_template_detail (五元组 → full schema)
  4. Save schema to file (NOT into context)
  5. Search for field schema fragment (recursive — groups contain fields)
  6. x-component → symbol index → component source code locations
  7. Return field schema + code locations
"""

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(".deer-flow/schemas")
SANDBOX_OUTPUTS_SCHEMA_DIR = "schemas"  # relative to outputs dir
VIRTUAL_OUTPUTS_PREFIX = "/mnt/user-data/outputs"

# MCP tool name suffixes (the full name includes server prefix)
MCP_LOCATE_TEMPLATE = "ace_ai_locate_template"
MCP_GET_TEMPLATE_DETAIL = "ace_ai_get_last_template_detail"


# ---------------------------------------------------------------------------
# Pure helper functions (no MCP dependency, independently testable)
# ---------------------------------------------------------------------------


def build_config_dimension(dim: dict) -> dict:
    """Build config_dimension dict from locate_template result.

    The locate_template API returns string values; get_last_template_detail
    expects integer values.
    """
    return {
        "category_id": int(dim["category_id"]),
        "product_type": int(dim["product_type"]),
        "product_sub_type": int(dim.get("product_sub_type", 0)),
        "template_type": int(dim.get("template_type", 1)),
        "template_sub_type": int(dim.get("template_sub_type", 0)),
    }


def _schema_filename(config_dimension: dict) -> str:
    """Generate safe filename from config_dimension dict."""
    dim_str = json.dumps(config_dimension, sort_keys=True)
    return re.sub(r'[{}":, ]+', "_", dim_str).strip("_") + ".json"


def save_schema_to_file(
    config_dimension: dict, schema: dict, schema_dir: Path = SCHEMA_DIR
) -> Path:
    """Save full schema to file. Keep it out of LLM context.

    Returns the physical path to the saved file.
    """
    schema_dir.mkdir(parents=True, exist_ok=True)
    path = schema_dir / _schema_filename(config_dimension)
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Schema saved to %s (%d bytes)", path, path.stat().st_size)
    return path


def search_field_in_schema(schema_config: dict, field_name: str) -> dict | None:
    """Search for a field in template schema_config by title, key, or bind_attr_keys.

    schema_config has nested structure: top-level properties are groups
    (basicInfo, merchantInfo, etc.), each group has its own properties
    containing the actual fields.

    Returns {"key": ..., "group": ..., "schema": ...} or None.
    """
    groups = schema_config.get("properties", {})
    if not groups:
        return None

    # Collect all fields from all groups
    all_fields: list[tuple[str, str, dict]] = []  # (group_key, field_key, field_schema)
    for group_key, group_schema in groups.items():
        group_props = group_schema.get("properties", {})
        for field_key, field_schema in group_props.items():
            all_fields.append((group_key, field_key, field_schema))

    # 1. Exact match by title
    for group_key, field_key, field_schema in all_fields:
        if field_schema.get("title") == field_name:
            return {"key": field_key, "group": group_key, "schema": field_schema}

    # 2. Match by field_key (camelCase key like "accountName")
    for group_key, field_key, field_schema in all_fields:
        if field_key == field_name:
            return {"key": field_key, "group": group_key, "schema": field_schema}

    # 3. Match by bind_attr_keys
    for group_key, field_key, field_schema in all_fields:
        attr_scope = field_schema.get("attr_keys_scope", {})
        bind_keys = attr_scope.get("bind_attr_keys", [])
        if field_name in bind_keys:
            return {"key": field_key, "group": group_key, "schema": field_schema}

    # 4. Match by x-component name
    for group_key, field_key, field_schema in all_fields:
        if field_schema.get("x-component") == field_name:
            return {"key": field_key, "group": group_key, "schema": field_schema}

    # 5. Fuzzy match by title containing field_name
    for group_key, field_key, field_schema in all_fields:
        title = field_schema.get("title", "")
        if title and field_name and (field_name in title or title in field_name):
            return {"key": field_key, "group": group_key, "schema": field_schema}

    return None


def list_all_fields(schema_config: dict) -> list[dict]:
    """List all fields from all groups in schema_config."""
    fields = []
    groups = schema_config.get("properties", {})
    for group_key, group_schema in groups.items():
        group_label = group_schema.get("x-component-props", {}).get("label", group_key)
        for field_key, field_schema in group_schema.get("properties", {}).items():
            fields.append({
                "group": group_label,
                "key": field_key,
                "title": field_schema.get("title", ""),
            })
    return fields


def extract_category_levels(category_full_name: str) -> list[str]:
    """Split '购物>果蔬生鲜>水果' into ['购物>果蔬生鲜>水果', '购物>果蔬生鲜', '购物'].

    Returns from most specific (leaf) to least specific (root).
    """
    parts = [p.strip() for p in category_full_name.split(">")]
    levels = []
    for i in range(len(parts), 0, -1):
        levels.append(">".join(parts[:i]))
    return levels


def locate_component_code(component_name: str, repo_name: str = "tobias-goods-mono") -> tuple[str, list[dict]]:
    """Search symbol index for component source code locations.

    Returns (repo_root_path, results) where results is list of
    {name, kind, file, line, span, exported} dicts.
    Empty list if index unavailable or no matches.
    """
    try:
        from scripts.index_repo import load_index, search_index
    except ImportError:
        logger.warning("index_repo not available, skipping component code lookup")
        return "", []

    index = load_index(repo_name)
    if not index:
        logger.info("No symbol index for repo '%s'", repo_name)
        return "", []

    results = search_index(index, component_name, limit=10)
    # Prefer component definitions (const/fn) over type definitions
    components = [r for r in results if r["kind"] in ("const/fn", "function", "class")]
    types = [r for r in results if r["kind"] in ("type", "interface")]
    return index.root_path, components + types


def extract_field_summary(field_schema: dict) -> dict:
    """Extract the decision-relevant fields from a field schema fragment.

    Returns only what LLM needs for analysis — not the full schema.
    """
    return {
        "title": field_schema.get("title"),
        "type": field_schema.get("type"),
        "x_component": field_schema.get("x-component"),
        "x_component_props": field_schema.get("x-component-props", {}),
        "required": field_schema.get("required", False),
        "x_disabled": field_schema.get("x-disabled", False),
        "default": field_schema.get("default"),
        "reaction_rules": field_schema.get("reaction_rules", []),
        "validator_rules": field_schema.get("validator_rules", []),
        "transfer_rule": field_schema.get("transfer_rule"),
        "attr_keys_scope": field_schema.get("attr_keys_scope"),
    }


# ---------------------------------------------------------------------------
# MCP tool resolution
# ---------------------------------------------------------------------------


def resolve_mcp_tool(mcp_tools: Mapping[str, Any], suffix: str) -> Any | None:
    """Find an MCP tool by name suffix (server prefix varies by config)."""
    for name, tool in mcp_tools.items():
        if name.endswith(suffix):
            return tool
    return None


def parse_mcp_response(result: Any) -> Any:
    """Extract data from MCP tool response.

    MCP tools return list[{"type":"text","text":"..."}]. Extract the text
    field and parse as JSON.
    """
    # list[{"type": "text", "text": "..."}] format from langchain-mcp-adapters
    if isinstance(result, list) and result:
        text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
    elif isinstance(result, str):
        text = result
    else:
        return result

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def get_mcp_tools_from_runtime() -> dict[str, BaseTool]:
    """Lazy-load MCP tools from DeerFlow runtime.

    Called only when the tool is actually invoked, not at registration time.
    """
    try:
        from deerflow.mcp.cache import get_cached_mcp_tools
        tools = get_cached_mcp_tools()
        return {t.name: t for t in (tools or [])}
    except ImportError:
        logger.warning("DeerFlow MCP module not available")
        return {}
    except Exception as e:
        logger.error("Failed to load MCP tools: %s", e)
        return {}


# ---------------------------------------------------------------------------
# The Programmatic Tool
# ---------------------------------------------------------------------------


class SchemaLocatorInput(BaseModel):
    """Input schema for locate_field_schema tool."""

    category_full_name: str = Field(
        description="类目路径，格式 '一级>二级>三级'，如 '购物>果蔬生鲜>水果'。至少二级。"
    )
    field_name: str = Field(
        default="",
        description="要查找的字段名称（标题、camelCase key、attr_key、组件名均可）。"
        "不传则返回模板所有字段列表概览。",
    )
    product_type: str = Field(
        default="",
        description="商品类型的数字枚举值。'1'=团购, '11'=代金券, '14'=配送商品, '15'=次卡 等。"
        "必须传数字字符串，不要传中文名称。完整枚举见 template_properties_map.json。",
    )
    product_sub_type: str = Field(
        default="",
        description="商品子类型的数字枚举值。'101'=权益卡团购, '1501'=周期卡 等。"
        "比 product_type 更精确，优先使用。不确定时不传。",
    )


class SchemaLocatorTool(BaseTool):
    """Locate template schema and extract field schema fragment.

    Programmatic tool: chains MCP calls without LLM in the loop.
    Category tree fallback: if leaf category has no template, walks up to parent/grandparent.

    Use when: user reports a form field issue (invisible, missing options, wrong validation)
    Don't use when: user asks about field VALUE errors or data issues (not schema/config problems)
    """

    name: str = "locate_field_schema"
    description: str = (
        "定位商品模板 schema 中的字段配置。输入类目路径和商品类型（数字枚举），"
        "返回字段的 schema 片段（显隐条件、校验规则、绑定组件等）。"
        "支持类目树向上回溯。\n\n"
        "Use when: 用户反馈表单字段不显示、缺少选项、校验异常等 schema 配置问题\n"
        "Don't use when: 用户反馈字段值错误、数据问题（非 schema/配置问题）"
    )
    args_schema: type[BaseModel] = SchemaLocatorInput

    schema_dir: str = str(SCHEMA_DIR)
    _mcp_tools: Mapping[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, mcp_tools: Mapping[str, Any] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._mcp_tools = mcp_tools

    def _get_mcp_tools(self) -> Mapping[str, Any]:
        if self._mcp_tools is None:
            self._mcp_tools = get_mcp_tools_from_runtime()
        return self._mcp_tools

    def _call_mcp(self, suffix: str, args: dict) -> Any:
        """Call an MCP tool by name suffix. Returns parsed data."""
        mcp_tools = self._get_mcp_tools()
        tool = resolve_mcp_tool(mcp_tools, suffix)
        if not tool:
            available = list(mcp_tools.keys())
            raise ToolException(
                f"MCP tool '*{suffix}' not found. Available: {available[:10]}. "
                "Check mcp_servers config in agent.yaml."
            )
        result = tool.invoke(args)
        return parse_mcp_response(result)

    def _locate_templates(
        self, category_full_name: str, product_type: str | None = None
    ) -> list[dict]:
        """Call locate_template MCP. Only pass non-empty params."""
        params: dict[str, str] = {"category_full_name": category_full_name}
        if product_type:
            params["product_type"] = product_type

        results = self._call_mcp(MCP_LOCATE_TEMPLATE, params)
        if not isinstance(results, list):
            return [results] if results else []
        return results

    def _locate_with_category_fallback(
        self, category_full_name: str, product_type: str | None = None
    ) -> list[dict]:
        """Walk up category tree to find the template.

        Given '购物>果蔬生鲜>水果', tries:
          1. '购物>果蔬生鲜>水果' (leaf)
          2. '购物>果蔬生鲜' (parent)
          3. '购物' (root)
        """
        levels = extract_category_levels(category_full_name)

        for level_path in levels:
            logger.info("Category fallback: trying '%s'", level_path)
            results = self._locate_templates(level_path, product_type)
            if results:
                logger.info("Found template at category level: %s (%d results)", level_path, len(results))
                return results

        return []

    def _get_template_detail(self, config_dim: dict) -> dict | None:
        """Call get_last_template_detail and extract schema_config."""
        result = self._call_mcp(MCP_GET_TEMPLATE_DETAIL, {
            "config_dimension": build_config_dimension(config_dim),
        })

        if not isinstance(result, dict):
            return None

        templates = result.get("online_template", [])
        if not templates:
            return None

        # Use first template's schema_config
        tmpl = templates[0]
        schema_config = tmpl.get("schema_config")
        if isinstance(schema_config, str):
            try:
                schema_config = json.loads(schema_config)
            except json.JSONDecodeError:
                return None
        return schema_config

    def _resolve_schema_dir(self) -> tuple[Path, str | None]:
        """Resolve where to save schema and the virtual path prefix.

        Returns (physical_dir, virtual_prefix_or_none).
        If thread_id is available, saves to sandbox outputs dir so sub-agents
        can access via read_file("/mnt/user-data/outputs/schemas/...").

        Uses DeerFlow's Paths system so the physical path matches what the
        sandbox read_file tool resolves for /mnt/user-data/outputs/.
        """
        # Get thread_id from LangGraph's context variable (set during agent execution)
        thread_id = None
        try:
            from langchain_core.runnables import ensure_config
            config = ensure_config()
            thread_id = config.get("configurable", {}).get("thread_id")
        except Exception:
            pass

        if thread_id:
            try:
                from deerflow.config.paths import get_paths
                outputs_dir = get_paths().sandbox_outputs_dir(thread_id)
            except ImportError:
                # Fallback if deerflow not available (shouldn't happen in agent runtime)
                outputs_dir = Path(f".deer-flow/threads/{thread_id}/user-data/outputs")
            schema_dir = outputs_dir / SANDBOX_OUTPUTS_SCHEMA_DIR
            virtual_prefix = f"{VIRTUAL_OUTPUTS_PREFIX}/{SANDBOX_OUTPUTS_SCHEMA_DIR}"
            return schema_dir, virtual_prefix

        # Fallback: no thread context (testing, standalone)
        return Path(self.schema_dir), None

    def _run(
        self,
        category_full_name: str,
        field_name: str = "",
        product_type: str = "",
        product_sub_type: str = "",
    ) -> dict:
        """Execute the Programmatic tool chain.

        Args:
            category_full_name: Category path like '购物>果蔬生鲜>水果'
            field_name: Field title, key, or attr_key to search for. Optional —
                if omitted, returns field list overview + schema_path.
            product_type: Product type string (e.g. '1' for 团购), optional
            product_sub_type: Product sub type string, optional
            run_manager: Injected by LangChain, provides access to thread config.
        """
        # --- Step 1: Locate template via MCP (with category fallback) ---
        templates = self._locate_with_category_fallback(
            category_full_name, product_type or None
        )

        if not templates:
            return {
                "status": "not_found",
                "error": f"未找到匹配 '{category_full_name}' 的模板，类目树全部层级均无匹配",
                "suggestion": "请确认类目名称和商品类型是否正确",
            }

        # --- Step 2: Extract config_dimension from response ---
        # locate_template returns [{"config_dimension": {...}}, ...]
        dimensions = []
        for t in templates:
            dim = t.get("config_dimension", t)  # unwrap if wrapped
            dimensions.append(dim)

        # --- Step 3: Disambiguation if multiple ---
        if len(dimensions) > 1:
            return {
                "status": "ambiguous",
                "candidates": [
                    {
                        "category_full_name": d.get("category_full_name", ""),
                        "category_id": d.get("category_id"),
                        "product_type": d.get("product_type"),
                        "product_sub_type": d.get("product_sub_type"),
                    }
                    for d in dimensions
                ],
                "hint": "找到多个匹配模板，请让用户确认具体是哪一个",
            }

        dim = dimensions[0]

        # --- Step 4: Get full schema via MCP ---
        schema_config = self._get_template_detail(dim)

        if not schema_config:
            return {
                "status": "schema_error",
                "config_dimension": build_config_dimension(dim),
                "error": "获取模板 schema_config 失败",
            }

        # --- Step 5: Save to file (NOT into context) ---
        config_dim_dict = build_config_dimension(dim)
        physical_dir, virtual_prefix = self._resolve_schema_dir()
        physical_path = save_schema_to_file(config_dim_dict, schema_config, physical_dir)
        filename = physical_path.name

        # schema_path for LLM/sub-agent: virtual if in sandbox, physical otherwise
        if virtual_prefix:
            schema_path = f"{virtual_prefix}/{filename}"
        else:
            schema_path = str(physical_path)

        # --- Step 6a: No field_name → return overview ---
        if not field_name:
            all_fields = list_all_fields(schema_config)
            return {
                "status": "overview",
                "config_dimension": config_dim_dict,
                "schema_path": schema_path,
                "category": dim.get("category_full_name", ""),
                "field_count": len(all_fields),
                "fields": all_fields,
                "hint": "全量 schema 已存到 schema_path，sub-agent 可通过 read_file 访问",
            }

        # --- Step 6b: Search for specific field ---
        field_result = search_field_in_schema(schema_config, field_name)

        if not field_result:
            all_fields = list_all_fields(schema_config)
            return {
                "status": "field_not_found",
                "config_dimension": config_dim_dict,
                "schema_path": schema_path,
                "error": f"在模板 schema 中未找到字段 '{field_name}'",
                "available_fields": all_fields[:30],
                "suggestion": f"sub-agent 可 read_file({schema_path}) 查找近似字段",
            }

        # --- Step 7: Locate component source code via symbol index ---
        summary = extract_field_summary(field_result["schema"])
        component_name = summary.get("x_component")
        repo_root = ""
        code_results: list[dict] = []
        if component_name:
            repo_root, code_results = locate_component_code(component_name)

        result = {
            "status": "found",
            "config_dimension": config_dim_dict,
            "schema_path": schema_path,
            "field_key": field_result["key"],
            "group": field_result["group"],
            "category": dim.get("category_full_name", ""),
            **summary,
        }

        if code_results:
            result["component_sources"] = [
                {
                    "file": c["file"],
                    "line": c["line"],
                    "span": c["span"],
                    "kind": c["kind"],
                    "exported": c["exported"],
                }
                for c in code_results
            ]
            if repo_root:
                result["code_repo_root"] = repo_root

        # --- Step 8: Build sub-agent prompt material ---
        import json as _json
        schema_summary = _json.dumps({
            "field_key": field_result["key"],
            "group": field_result["group"],
            **summary,
        }, ensure_ascii=False, indent=2)

        # Build bash commands for reading component source code
        # Symbol index has line + span → read only the definition, not the whole file.
        # Hooks/utils are typically small, so cat the whole file.
        bash_commands = []
        if code_results and repo_root:
            impl_files = [c for c in code_results if c["kind"] in ("const/fn", "function", "class")]
            seen_dirs: set[str] = set()
            for c in impl_files[:3]:
                full_path = f"{repo_root}/{c['file']}"
                start = max(1, c["line"] - 5)  # 5 lines context before
                end = c["line"] + c["span"] + 5  # 5 lines context after
                bash_commands.append(
                    f'bash("读取 {c["file"]}:{c["line"]} 的组件定义", '
                    f'"sed -n \'{start},{end}p\' {full_path}")'
                )
                # Scan sibling hooks/utils — key business logic lives here
                comp_dir = Path(full_path).parent
                if str(comp_dir) not in seen_dirs:
                    seen_dirs.add(str(comp_dir))
                    for subdir in ("hooks", "utils"):
                        sub_path = comp_dir / subdir
                        if sub_path.is_dir():
                            for f in sorted(sub_path.glob("*.ts"))[:3]:
                                bash_commands.append(f'bash("读取 {f.name}", "cat {f}")')
                            for f in sorted(sub_path.glob("*.tsx"))[:3]:
                                bash_commands.append(f'bash("读取 {f.name}", "cat {f}")')

        result["next_action"] = (
            "请使用 task 工具委派 sub-agent 分析。将以下内容完整复制到 sub-agent prompt 中：\n\n"
            "--- sub-agent prompt 开始 ---\n"
            "## 任务\n"
            "分析用户反馈的字段问题，给出根因和解决建议。\n\n"
        )

        # Step 1: Read source code FIRST (if available)
        if bash_commands:
            result["next_action"] += (
                "## 第一步：读取组件源码（必须执行，不可跳过）\n"
                "依次执行以下命令，读取源码后再进行分析：\n"
                + "\n".join(bash_commands) + "\n\n"
            )

        # Step 2: Schema data for reference
        result["next_action"] += (
            "## 第二步：结合 schema 配置分析\n"
            "### 字段 schema 配置（已提取，无需再查）\n"
            f"```json\n{schema_summary}\n```\n\n"
            f"全量 schema 文件（必要时 read_file 查看关联字段）：{schema_path}\n\n"
            "### 分析要点\n"
            "- 分析 reaction_rules、x-hidden、required、分组条件\n"
            "- 结合第一步读到的组件源码，分析运行时显隐逻辑\n"
            "- 综合 schema 配置 + 代码运行时逻辑 给出根因\n\n"
        )

        result["next_action"] += (
            "## 禁止事项\n"
            "- 不要调用 locate_field_schema（已经查过了）\n"
            "- 不要跳过第一步直接分析（没有源码的分析是不完整的）\n"
            "--- sub-agent prompt 结束 ---"
        )

        return result


# Module-level instance for config.yaml registration
locate_field_schema_tool = SchemaLocatorTool()
