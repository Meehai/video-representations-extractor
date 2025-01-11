import numpy as np
import pytest
from vre.representations.build_representations import build_representation_from_cfg
from vre.representations import ComputeRepresentationMixin, LearnedRepresentationMixin, IORepresentationMixin
from vre.utils import parsed_str_type
from vre_repository import get_vre_repository as greps

def test_build_representations_from_cfg_defaults():
    orig_cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(orig_cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert parsed_str_type(res) == "RGB" and isinstance(res, ComputeRepresentationMixin), res
    assert not isinstance(res, LearnedRepresentationMixin) and isinstance(res, IORepresentationMixin), res
    assert res.batch_size == 1 and res.output_dtype is np.dtype("uint8") and res.output_size is "video_shape"

def test_build_representations_from_cfg_device():
    base_cfg = {"dependencies": [], "parameters": {}}

    cfg = {**base_cfg, "type": "edges/dexined"}
    res = build_representation_from_cfg(cfg, name="dexined", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert parsed_str_type(res) == "DexiNed" and isinstance(res, ComputeRepresentationMixin)
    assert isinstance(res, LearnedRepresentationMixin), res
    assert res.device == "cpu"

    cfg = {**base_cfg, "type": "edges/dexined"}
    res = build_representation_from_cfg(cfg, name="dexined", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={"device": "cpuXX"}, io_defaults={})
    assert res.device == "cpuXX"

    cfg = {**base_cfg, "type": "edges/dexined", "learned_parameters": {"device": "cuda"}}
    res = build_representation_from_cfg(cfg, name="dexined", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.device == "cuda"

    cfg = {**base_cfg, "type": "edges/dexined", "learned_parameters": {"device": "cuda"}}
    res = build_representation_from_cfg(cfg, name="dexined", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={"device": "cpuXX"}, io_defaults={})
    assert res.device == "cuda"

def test_build_representations_from_cfg_batch_size():
    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.batch_size == 1, res.batch_size

    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={"batch_size": 10}, learned_defaults={}, io_defaults={})
    assert res.batch_size == 10, res.batch_size

    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.batch_size == 15, res.batch_size

    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={"batch_size": 10}, learned_defaults={}, io_defaults={})
    assert res.batch_size == 15, res.batch_size

    with pytest.raises(AssertionError):
        cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": 15.5}}
        _ = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})
    with pytest.raises(AssertionError):
        cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "compute_parameters": {"batch_size": "ABC"}}
        _ = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})

def test_build_representations_from_cfg_output_dtype():
    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.output_dtype == np.dtype("uint8"), res.output_dtype

    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={},
                                        io_defaults={"output_dtype": "uint16"})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "io_parameters": {"output_dtype": "uint16"}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "io_parameters": {"output_dtype": "uint16"}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={"output_dtype": "uint8"})
    assert res.output_dtype == np.dtype("uint16"), res.output_dtype

    with pytest.raises(TypeError):
        cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "io_parameters": {"output_dtype": "lala"}}
        _ = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={},
                                          io_defaults={"output_dtype": "lala"})

    with pytest.raises(TypeError):
        cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}, "io_parameters": {"output_dtype": "lala"}}
        _ = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={},
                                          io_defaults={"output_dtype": "float16"})

    cfg = {"type": "edges/dexined", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(cfg, name="dexined", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert res.output_dtype == np.dtype("float32"), res.output_dtype

def test_build_representations_from_cfg_output_size():
    orig_cfg = {"type": "color/rgb", "dependencies": [], "parameters": {}}
    res = build_representation_from_cfg(orig_cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert isinstance(res, IORepresentationMixin), res
    assert res.output_size == "video_shape", res.output_size

    res = build_representation_from_cfg(orig_cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={"output_size": "native"})
    assert isinstance(res, IORepresentationMixin), res
    assert res.output_size == "native", res.output_size

    cfg = {**orig_cfg, "io_parameters": {"output_size": [100, 200]}}
    res: IORepresentationMixin = build_representation_from_cfg(
        cfg, name="rgb", representation_types=greps(), built_so_far={},
        compute_defaults={}, learned_defaults={}, io_defaults={})
    assert isinstance(res, IORepresentationMixin), res
    assert res.output_size == (100, 200), res.output_size

    cfg = {**orig_cfg, "io_parameters": {"output_size": [100, 200]}}
    res = build_representation_from_cfg(cfg, name="rgb", representation_types=greps(), built_so_far={},
                                        compute_defaults={}, learned_defaults={}, io_defaults={"output_size": "native"})
    assert isinstance(res, IORepresentationMixin), res
    assert res.output_size == (100, 200), res.output_size

    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "io_parameters": {"output_size": "lala"}},
                                          name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "io_parameters": {"output_size": [100, 200, 300]}},
                                          name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "io_parameters": {"output_size": [100, 200.5]}},
                                          name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})
    with pytest.raises(AssertionError):
        _ = build_representation_from_cfg({**orig_cfg, "io_parameters": {"output_size": [-15, 100]}},
                                          name="rgb", representation_types=greps(), built_so_far={},
                                          compute_defaults={}, learned_defaults={}, io_defaults={})
