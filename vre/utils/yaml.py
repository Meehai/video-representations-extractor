"""
yaml utils -- a minimum subset of stuff from omegaconf
Warning! This yaml syntax is not supported:
key: {other_key: ${oc.env}}
Use this instead:
key:
  other_key: ${oc.env}
"""
from io import IOBase, StringIO
from pathlib import Path
import ast
import os
from typing import Any
import yaml

def _parse_atom(v: list | dict | str | int | float | bool) -> list | dict | str | int | float | bool:
    """parses the config file and resolves environment variables or decodes them if needed. Emulates omegaconf."""
    assert isinstance(v, (list, dict, str, int, float, bool)), type(v)
    if isinstance(v, list):
        return [_parse_atom(_v) for _v in v]
    if isinstance(v, dict):
        return {k: _parse_atom(_v) for k, _v in v.items()}
    if not isinstance(v, str):
        return v
    if v.startswith("${oc.env"):
        assert v.find("{", 9) == -1, f"cannot have embedded oc inside oc.env: {v}"
        assert v[-1] == "}", f"doesn't end in '}}': {v}"
        key_and_default = v[9:-1].split(",")
        assert len(key_and_default) <= 2, (v, key_and_default)
        key, default = key_and_default[0], None if len(key_and_default) == 1 else key_and_default[1]
        if default is None and key not in os.environ:
            raise KeyError(f"Environment variable: '{key}' not set and no default provided.")
        value = os.getenv(key, default)
        return value
    if v.startswith("${oc.decode"):
        inner = _parse_atom(v[12:-1])
        return ast.literal_eval(inner)
    return v

def vre_yaml_load(path: Path | str | IOBase) -> dict[str, Any]:
    """reads a yaml file and resolves the env varaibles if needed"""
    fp = open(path, "r") if isinstance(path, (Path, str)) else path
    fp.seek(0)
    data: str = fp.read()
    assert len(data) > 0, f"Nothing read from '{path}'"

    cfg_raw = yaml.safe_load(StringIO(data))
    cfg = _parse_atom(cfg_raw)
    fp.close() if isinstance(path, (Path, str)) else None
    return cfg
