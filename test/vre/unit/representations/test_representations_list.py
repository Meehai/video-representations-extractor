import sys
from vre.utils import get_project_root
from vre.representations import RepresentationsList
import pytest

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

def test_RepresentationsList_basic():
    repr_list = RepresentationsList([
        FakeRepresentation("r1"),
        FakeRepresentation("r2")
    ])

    assert len(repr_list) == 2 and set(repr_list.names) == {"r1", "r2"}

def test_RepresentationsList_toposort_fail_1():
    with pytest.raises(ValueError):
        _ = RepresentationsList([
            FakeRepresentation("r1"),
            FakeRepresentation("r2"),
            FakeRepresentation("r1"),
        ])

def test_RepresentationsList_toposort_fail_2():
    r1 = FakeRepresentation("r1")
    r2 = FakeRepresentation("r2")
    r3 = FakeRepresentation("r3")
    r1.dependencies = [r2]
    r2.dependencies = [r3]
    r3.dependencies = [r1]

    with pytest.raises(AssertionError):
        _ = RepresentationsList([
            r1,
            r2,
            r3,
        ])

def test_RepresentationsList_get_exported_representations():
    repr_list = RepresentationsList([
        r1 := FakeRepresentation("r1"),
        r2 := FakeRepresentation("r2")
    ])
    r1.binary_format = "npz"
    assert repr_list.get_output_representations().names == ["r1"]
    r2.binary_format = "npz"
    assert set(repr_list.get_output_representations().names) == {"r1", "r2"}
    assert repr_list.get_output_representations(subset=["r1"]).names == ["r1"]

def test_RepresentationsList_get_exported_representations_2():
    # r1 <- r2 <- r3 and i want just r1 and r3
    repr_list = RepresentationsList([
        r1 := FakeRepresentation("r1"),
        r2 := FakeRepresentation("r2", dependencies=[r1]),
        r3 := FakeRepresentation("r3", dependencies=[r2]),
    ])
    r1.binary_format = "npz"
    # r2.binary_format = "npz"
    r3.binary_format = "npz"
    assert repr_list.get_output_representations().names == ["r1", "r3"]

def test_RepresentationsList_get_exported_representations_3():
    # r1 <- r2 <- r3 and i want just r1 and r3
    repr_list = RepresentationsList([
        r1 := FakeRepresentation("r1"),
        r2 := FakeRepresentation("r2", dependencies=[r1]),
        r3 := FakeRepresentation("r3", dependencies=[r2]),
    ])
    r1.binary_format = "npz"
    r2.binary_format = "npz"
    r3.binary_format = "npz"
    assert repr_list.get_output_representations().names == ["r1", "r2", "r3"]
    assert repr_list.get_output_representations(["r1", "r3"]).names == ["r1", "r3"]
