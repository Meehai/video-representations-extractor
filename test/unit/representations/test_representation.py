from vre.representations import FakeRepresentation
from vre.utils import FakeVideo
import numpy as np
import pytest

class SimpleRepresentation(FakeRepresentation): pass

@pytest.fixture
def video() -> FakeVideo:
    return FakeVideo(np.random.randint(0, 255, size=(10, 20, 30, 3), dtype=np.uint8), frame_rate=25)

def test_representation_ctor():
    repr = SimpleRepresentation("simple_representation")
    assert repr.name == "simple_representation"

def test_representation_make(video):
    repr = SimpleRepresentation("simple_representation")
    repr.video = video
    assert np.allclose(repr.make(video[0:5]).output, video[0:5])
