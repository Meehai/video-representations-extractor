import sys
from pathlib import Path
import pytest
from vre.utils import get_project_root, parsed_str_type
from vre.representations import add_external_repositories, build_representations_from_cfg

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

class MyRepresentation(FakeRepresentation): pass
class MyOherRepresentation(FakeRepresentation): pass
class MyOherRepresentation2(FakeRepresentation): pass

def get_representations_1():
    return {
        "1/my_representation": MyRepresentation,
        "1/my_other_representation": MyOherRepresentation
    }

def get_representations_1_clash():
    return {
        "1/my_representation_asdf": MyRepresentation,
        "1/my_other_representation": MyOherRepresentation2
    }

def get_representations_2():
    return {
        "2/my_representation": MyRepresentation,
        "2/my_other_representation": MyOherRepresentation
    }

def test_add_external_representations():
    with pytest.raises(AssertionError):
        _ = add_external_repositories([f"{Path(__file__)}:get_representations_1",
                                       f"{Path(__file__)}:get_representations_1_clash"])

    repr_types_1 = add_external_repositories([f"{Path(__file__)}:get_representations_1"])
    assert len(repr_types_1) == 2, repr_types_1
    cfg = {
        "representations": {
            "my_rep1": {"type": "1/my_representation", "dependencies": [], "parameters": {}},
            "my_rep2": {"type": "2/my_other_representation", "dependencies": ["my_rep1"], "parameters": {}},
        }
    }
    with pytest.raises(KeyError):
        _ = build_representations_from_cfg(cfg, repr_types_1)
    repr_types_2 = add_external_repositories([f"{Path(__file__)}:get_representations_2"], repr_types_1)
    reps = build_representations_from_cfg(cfg, repr_types_2)
    assert len(reps) == 2
    assert parsed_str_type(reps[0]) == "MyRepresentation", type(reps[0])
    assert parsed_str_type(reps[1]) == "MyOherRepresentation", type(reps[1])
    assert reps[1].dependencies[0] == reps[0] and len(reps[1].dependencies) == 1, reps[1].dependencies
