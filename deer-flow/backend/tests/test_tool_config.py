import pydantic
import pytest

from deerflow.config.tool_config import ToolConfig


def test_tool_config_defaults_ptc_eligible_to_false():
    """ptc_eligible defaults to False when not specified."""
    cfg = ToolConfig(name="bash", group="sandbox", use="deerflow.sandbox.tools:bash_tool")
    assert cfg.ptc_eligible is False


def test_tool_config_accepts_ptc_eligible_true():
    """ptc_eligible can be explicitly set to True."""
    cfg = ToolConfig(
        name="bash",
        group="sandbox",
        use="deerflow.sandbox.tools:bash_tool",
        ptc_eligible=True,
    )
    assert cfg.ptc_eligible is True


def test_tool_config_rejects_invalid_ptc_eligible_type():
    """ptc_eligible only accepts bool-compatible values."""
    with pytest.raises(pydantic.ValidationError):
        ToolConfig(
            name="bash",
            group="sandbox",
            use="deerflow.sandbox.tools:bash_tool",
            ptc_eligible="not-a-bool",  # type: ignore
        )
