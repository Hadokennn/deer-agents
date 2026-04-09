"""Built-in subagent configurations."""

from .bash_agent import BASH_AGENT_CONFIG
from .code_analyst import CODE_ANALYST_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "CODE_ANALYST_CONFIG",
]

# Registry of built-in subagents
BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
    "code-analyst": CODE_ANALYST_CONFIG,
}
