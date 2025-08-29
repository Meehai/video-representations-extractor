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

def _parse(data: str) -> str:
    if (ix := data.find("${")) == -1:
        return data
    left, current = data[0:ix], data[ix:]
    if current.startswith("${oc.env:"):
        current = _parse(current[len("${oc.env:"):]) # recursively resolve inner oc.env:{} stuff
        assert (ix_r := current.find("}")) != -1, current
        key_and_default = current[0:ix_r].split(",")
        assert len(key_and_default) >= 1, current
        key, default = key_and_default[0], None
        if len(key_and_default):
            default = ",".join(key_and_default[1:]) # in case we have list defaults
        if len(key_and_default) == 1 and key not in os.environ:
            raise KeyError(f"Environment variable: '{key}' not set and no default provided.")
        value = os.getenv(key, default)
        current = f"{value}{current[ix_r+1:]}" # get rid of the '}' of this oc.env and replace it with the env value
    if current.startswith("${oc.decode:"):
        current = _parse(current[len("${oc.decode:"):]) # recursively resolve inner oc.env:{} stuff
        assert (ix_r := current.find("}")) != -1, current
        value = ast.literal_eval(current[0:ix_r])
        current = f"{value}{current[ix_r+1:]}" # get rid of the '}' of this oc.decode and replace it with the decoded
    return f"{left}{current}"

def vre_yaml_load(path: Path | str | IOBase) -> dict[str, Any]:
    """reads a yaml file and resolves the env varaibles if needed"""
    fp = open(path, "r") if isinstance(path, (Path, str)) else path
    fp.seek(0)
    data: str = fp.read()
    assert len(data) > 0, f"Nothing read from '{path}'"

    cfg_str = _parse(data)
    try:
        cfg = yaml.safe_load(StringIO(cfg_str))
    except Exception as e:
        print(f"-Could not parse:\n{cfg_str}\n-Exception: {e}")
        raise e
    fp.close() if isinstance(path, (Path, str)) else None
    return cfg
