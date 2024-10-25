from vre.representations.build_representations import build_representation_type, build_representation_from_cfg
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

def test_build_representations_from_cfg():
    orig_cfg = {"type": "default/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(orig_cfg, name="rgb", built_so_far={})
    assert parsed_str_type(res) == "RGB"
    assert res.batch_size is None and res.output_dtype is None and res.output_size is None

    cfg = {**orig_cfg, "batch_size": 5, "device": "cuda", "output_dtype": "uint8"}
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    
    cfg = {**orig_cfg, "batch_size": 5, "output_dtype": "lala"}
    with pytest.raises(TypeError):
        _ = build_representation_from_cfg(cfg, name="rgb", built_so_far={})

    cfg = {**orig_cfg, "batch_size": 5, "output_dtype": "uint16"}
    res = build_representation_from_cfg(cfg, name="rgb", built_so_far={})
    assert parsed_str_type(res) == "RGB"
    assert res.batch_size == 5 and res.output_dtype == np.dtype("uint16") and res.output_size is None

    obj = build_representation_from_cfg({**orig_cfg, "output_size": "video_shape"}, name="rgb", built_so_far={})
    assert obj.output_size == "video_shape"
    obj = build_representation_from_cfg({**orig_cfg, "output_size": "native"}, name="rgb", built_so_far={})
    assert obj.output_size == "native"
    obj = build_representation_from_cfg({**orig_cfg, "output_size": [100, 200]}, name="rgb", built_so_far={})
    assert obj.output_size == [100, 200]

    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "output_size": "lala"}, name="rgb", built_so_far={})
        _ = build_representation_from_cfg({**orig_cfg, "output_size": [100, 200, 300]}, name="rgb", built_so_far={})
        _ = build_representation_from_cfg({**orig_cfg, "output_size": [100, 200.5]}, name="rgb", built_so_far={})
        _ = build_representation_from_cfg({**orig_cfg, "output_size": [-15, 100]}, name="rgb", built_so_far={})
