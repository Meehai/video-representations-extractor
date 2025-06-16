import os
from io import StringIO
import pytest
from vre.utils import vre_yaml_load

def test_vre_yaml_load_one_env_var():
    cfg_str = StringIO("""
default_learned_parameters:
  device: ${oc.env:VRE_TEST_ENV,cpu}
  hi: 1
hi2: {gogu: talks}
""")
    res = vre_yaml_load(cfg_str)
    assert res == {
        "default_learned_parameters": {"device": "cpu", "hi": 1},
        "hi2": {"gogu": "talks"}
    }

def test_vre_yaml_load_bad_env_var():
    cfg_str = StringIO("""
default_learned_parameters:
  device: ${oc.env:{{,cpu}
""")
    with pytest.raises(ValueError):
        _ = vre_yaml_load(cfg_str)

    cfg_str = StringIO("""
default_learned_parameters:
  device: ${oc.env:HELLO
""")
    with pytest.raises(ValueError):
        _ = vre_yaml_load(cfg_str)

def test_vre_yaml_load_two_env_vars():
    cfg_str = StringIO("""
default_learned_parameters:
  device: ${oc.env:VRE_TEST_ENV,cpu}
  hi: 1
hi2: {gogu: ${oc.env:VRE_TEST_ENV2}}
""")
    with pytest.raises(KeyError):
        vre_yaml_load(cfg_str)

    for key in ["key1", "lalala"]:
        os.environ["VRE_TEST_ENV2"] = key
        res = vre_yaml_load(cfg_str)
        assert res == {
            "default_learned_parameters": {"device": "cpu", "hi": 1},
            "hi2": {"gogu": key}
        }
