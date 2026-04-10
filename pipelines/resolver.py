"""Variable resolution: ${var} template substitution."""

import re
from typing import Any

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()


class VariableResolver:
    """Resolves ${var} templates against an execution context."""

    def __init__(self, context: dict[str, Any]):
        self.context = context

    def resolve(self, value: Any) -> Any:
        """Recursively resolve templates in any value (str, dict, list, scalar)."""
        if isinstance(value, str):
            return self._resolve_string(value)
        if isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        return value

    def _resolve_string(self, s: str) -> Any:
        match = _TEMPLATE_PATTERN.fullmatch(s)
        if match:
            return self._eval_expression(match.group(1))

        def _replace(m: re.Match) -> str:
            val = self._eval_expression(m.group(1))
            return "" if val is None else str(val)

        return _TEMPLATE_PATTERN.sub(_replace, s)

    def _eval_expression(self, expr: str) -> Any:
        parts = [p.strip() for p in expr.split("|")]
        for part in parts:
            value = self._eval_atom(part)
            if value is not _MISSING and value is not None:
                return value
        return None

    def _eval_atom(self, atom: str) -> Any:
        if (atom.startswith('"') and atom.endswith('"')) or (
            atom.startswith("'") and atom.endswith("'")
        ):
            return atom[1:-1]
        return self._get_path(atom)

    def _get_path(self, path: str) -> Any:
        parts = path.split(".")
        current: Any = self.context
        for part in parts:
            if current is None:
                return _MISSING
            if isinstance(current, dict):
                if part not in current:
                    return _MISSING
                current = current[part]
            else:
                if not hasattr(current, part):
                    return _MISSING
                current = getattr(current, part)
        return current
