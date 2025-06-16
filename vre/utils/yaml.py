"""yaml utils -- a minimum subset of stuff from omegaconf"""
from io import IOBase, StringIO
from pathlib import Path
import os
from typing import Any
import yaml

def vre_yaml_load(path: Path | str | IOBase) -> dict[str, Any]:
    """reads a yaml file and resolves the env varaibles if needed"""
    fp = open(path, "r") if isinstance(path, (Path, str)) else path
    fp.seek(0)
    data: str = fp.read()
    assert len(data) > 0, f"Nothing read from '{path}'"
    skip = 0
    new_data = ""
    while (ix := data.find("${oc.env:", skip)) != -1:
        new_data += data[skip: ix]
        ix_r = data.find("}", ix+9)
        if ix_r == -1:
            raise ValueError(f"Unfinished oc.env: {data[ix:]}")
        key_and_default = data[ix+9: ix_r].split(",")
        assert len(key_and_default) <= 2, key_and_default
        has_default = False
        key, default = key_and_default[0], None
        if len(key_and_default) == 2:
            has_default = True
            key, default = key_and_default
        if not has_default and key not in os.environ:
            raise KeyError(f"Environment variable: '{key}' not set and no default provided.")
        if "{" in key:
            raise ValueError("'{' cannot be in the read key: '" + key + "'")
        read_key = os.getenv(key, default)
        new_data += read_key
        skip = ix_r + 1
    new_data += data[skip:]
    cfg = yaml.safe_load(StringIO(new_data))
    return cfg
