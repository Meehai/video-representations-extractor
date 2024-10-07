from vre.representations.build_representations import build_representation_type
import pytest

def test_build_representation_type():
    with pytest.raises(AssertionError):
        _ = build_representation_type("a/b/c")
    with pytest.raises(ValueError):
        _ = build_representation_type("UNKNOWN/representation")
    res = build_representation_type("depth/dpt")
    assert str(res).split("'")[1] == "vre.representations.depth.dpt.depth_dpt.DepthDpt"
