from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolGroupConfig(BaseModel):
    """Config section for a tool group"""

    name: str = Field(..., description="Unique name for the tool group")
    model_config = ConfigDict(extra="allow")


class ToolConfig(BaseModel):
    """Config section for a tool"""

    name: str = Field(..., description="Unique name for the tool")
    group: str = Field(..., description="Group name for the tool")
    use: str = Field(
        ...,
        description="Variable name of the tool provider(e.g. deerflow.sandbox.tools:bash_tool)",
    )
    model_config = ConfigDict(extra="allow")


class PTCEligibleToolConfig(BaseModel):
    """One tool that a purpose-scoped PTC tool can call.

    The tool is referenced by name; its implementation is looked up in the
    same tool registry used by get_available_tools (config tools, built-ins,
    MCP tools, ACP tools). `output_schema` is optional — when provided it
    overrides the tool's auto-detected output shape in the PTC wrapper's
    function-signature docs.
    """

    name: str = Field(..., description="Name of the tool as registered in the global tool registry")
    output_schema: dict | None = Field(
        default=None,
        description="Optional JSON Schema describing the tool's output. Used when the underlying tool does not declare one.",
    )

    model_config = ConfigDict(extra="forbid")


class PTCToolConfig(BaseModel):
    """Declaration of a purpose-scoped Programmatic Tool Calling (PTC) tool.

    A PTC tool is a LangChain tool that accepts a `code: str` argument,
    executes that code in a restricted Python namespace where each
    eligible tool is available as a callable, and returns the printed
    output. It is intended for focused, purpose-specific workflows.

    See docs/superpowers/specs/2026-04-13-purpose-scoped-ptc-design.md
    """

    name: str = Field(
        ...,
        description="Unique tool name. Must not collide with any other tool name in the system.",
    )
    purpose: str = Field(
        ...,
        description="Trigger-oriented description telling the LLM when to pick this tool and what it returns.",
    )
    eligible_tools: list[PTCEligibleToolConfig] = Field(
        ...,
        min_length=1,
        description="Tools the PTC code may call. Must contain at least one entry.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Max wall time for this PTC tool. None → use module default (30s).",
    )
    max_output_chars: int | None = Field(
        default=None,
        ge=100,
        description="Max stdout chars returned. None → use module default (20000).",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("eligible_tools")
    @classmethod
    def _no_empty_eligible(cls, v: list) -> list:
        if not v:
            raise ValueError("eligible_tools must contain at least one tool")
        return v
