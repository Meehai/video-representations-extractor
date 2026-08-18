"""
pytest conftest: install typeguard's import hook BEFORE any test imports image_utils.

Scoped to our own modules only. The hook rewrites annotated functions to check types at
runtime; with no scope it would also rewrite PIL/numpy, whose runtime annotations use
TYPE_CHECKING-only names (e.g. StrOrBytesPath) and blow up. The list covers both import
modes: top-level modules ("image_utils", "image_utils_pil") and the package submodules
("image_utils.image_utils*" via the startswith match).

conftest.py is imported during collection, before the test modules that import image_utils,
so the hook is in place in time to instrument our code (an in-module hook is too late).
"""
import os

if os.getenv("TYPEGUARD", "0") == "1":
    from typeguard import install_import_hook
    install_import_hook(["image_utils", "image_utils_pil"])
