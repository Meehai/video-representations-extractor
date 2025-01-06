import sys
from vre.utils import FakeVideo, get_project_root
import numpy as np
import pytest

sys.path.append(str(get_project_root() / "test"))
from fake_representation import FakeRepresentation

class SimpleRepresentation(FakeRepresentation): pass

@pytest.fixture
def video() -> FakeVideo:
    return FakeVideo(np.random.randint(0, 255, size=(10, 20, 30, 3), dtype=np.uint8), fps=25)

def test_representation_ctor():
    repr = SimpleRepresentation("simple_representation")
    assert repr.name == "simple_representation"

def test_representation_compute(video):
    repr = SimpleRepresentation("simple_representation")
    repr.compute(video, [0, 1, 2, 3, 4])
    assert np.allclose(repr.data.output, video[0:5])
