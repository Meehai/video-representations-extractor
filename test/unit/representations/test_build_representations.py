from vre.representations.build_representations import build_representation_type, build_representation_from_cfg
from vre.representations import ComputeRepresentationMixin, LearnedRepresentationMixin
from vre.utils import parsed_str_type
import numpy as np
import pytest

def test_build_representation_type():
    with pytest.raises(AssertionError):
        _ = build_representation_type("a/b/c")
    with pytest.raises(ValueError):
        _ = build_representation_type("UNKNOWN/representation")
    res = build_representation_type("depth/dpt")
    assert str(res).split("'")[1] == "vre.representations.depth.dpt.depth_dpt.DepthDpt"

def test_build_representations_from_cfg_defaults():
    orig_cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(orig_cfg, name="rgb", built_so_far={})
    assert parsed_str_type(res) == "RGB" and isinstance(res, ComputeRepresentationMixin), res
    assert not isinstance(res, LearnedRepresentationMixin), res
    assert res.batch_size == 1 and res.output_dtype is np.dtype("uint8") and res.output_size is "video_shape"

def test_build_representations_from_cfg_device():
    base_cfg = {"dependencies": [], "parameters": {}}

    cfg = {**base_cfg, "type": "edges/dexined"}
    res = build_representation_from_cfg(cfg, name="dexined", built_so_far={})
    assert parsed_str_type(res) == "DexiNed" and isinstance(res, ComputeRepresentationMixin)
    assert isinstance(res, LearnedRepresentationMixin), res
    assert res.device == "cpu"

    cfg = {**base_cfg, "type": "edges/dexined"}
    res = build_representation_from_cfg(cfg, name="dexined", built_so_far={},
                                        learned_representations_defaults={"device": "cpuXX"})
    assert res.device == "cpuXX"

    cfg = {**base_cfg, "type": "edges/dexined", "learned_parameters": {"device": "cuda"}}
    res = build_representation_from_cfg(cfg, name="dexined", built_so_far={})
    assert res.device == "cuda"

    cfg = {**base_cfg, "type": "edges/dexined", "learned_parameters": {"device": "cuda"}}
    res = build_representation_from_cfg(cfg, name="dexined", built_so_far={},
                                        learned_representations_defaults={"device": "cpuXX"})
    assert res.device == "cuda"

def test_build_representations_from_cfg_batch_size():
    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert res.batch_size == 1, res.batch_size

    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"batch_size": 10})
    assert res.batch_size == 10, res.batch_size

    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert res.batch_size == 15, res.batch_size

    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"batch_size": 10})
    assert res.batch_size == 15, res.batch_size

    with pytest.raises(AssertionError):
        cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15.5}}
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    with pytest.raises(AssertionError):
        cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": "ABC"}}
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={})

def test_build_representations_from_cfg_output_dtype():
    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert res.output_dtype == np.dtype("uint8"), res.output_dtype

    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"output_dtype": "uint16"})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {},
           "compute_parameters": {"output_dtype": "uint16"}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    cfg = {"type": "default/rgb", "dependencies": [], "parameters": {},
           "compute_parameters": {"output_dtype": "uint16"}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"output_dtype": "uint8"})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    with pytest.raises(TypeError):
        cfg = {"type": "default/rgb", "dependencies": [], "parameters": {},
               "compute_parameters": {"output_dtype": "lala"}}
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                          compute_representations_defaults={"output_dtype": "lala"})

    with pytest.raises(TypeError):
        cfg = {"type": "default/rgb", "dependencies": [], "parameters": {},
               "compute_parameters": {"output_dtype": "lala"}}
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                          compute_representations_defaults={"output_dtype": "float16"})

    cfg = {"type": "edges/dexined", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="dexined", built_so_far={})
    assert res.output_dtype == np.dtype("float32"), res.output_dtype

def test_build_representations_from_cfg_output_size():
    orig_cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(orig_cfg, name="rgb", built_so_far={})
    assert res.output_size == "video_shape", res.output_size

    res = build_representation_from_cfg(orig_cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"output_size": "native"})
    assert res.output_size == "native", res.output_size

    cfg = {**orig_cfg, "compute_parameters": {"output_size": [100, 200]}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert res.output_size == (100, 200), res.output_size

    cfg = {**orig_cfg, "compute_parameters": {"output_size": [100, 200]}}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={},
                                        compute_representations_defaults={"output_size": "native"})
    assert res.output_size == (100, 200), res.output_size

    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "compute_parameters": {"output_size": "lala"}},
                                          name="rgb", built_so_far={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "compute_parameters": {"output_size": [100, 200, 300]}},
                                          name="rgb", built_so_far={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "compute_parameters": {"output_size": [100, 200.5]}},
                                          name="rgb", built_so_far={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "compute_parameters": {"output_size": [-15, 100]}},
                                          name="rgb", built_so_far={})
