"""First-party micro-shim for the `@overrides` / `@override` decorator (existence check only).

Not the third-party `overrides` package — see README.md.
"""
from .overrides import overrides, override

__all__ = ["overrides", "override"]
