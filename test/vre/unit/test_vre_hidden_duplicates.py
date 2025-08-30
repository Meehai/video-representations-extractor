import sys
import numpy as np
import pytest
from vre.utils import get_project_root
from vre import VRE
from vre_video import VREVideo

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation

def test_vre_hidden_duplicates_1():
    """
    this test looks for "hidden" dupplicates: i.e. the dependencies of a representation differs from the main one
    For example, in this case:
    - repr1 <- dep1 (fake)
    - dep1

    We don't explicitly check that dep1 (main) and dep1 (dep) are the same object (or at least the same type/params)
    """

    dep1_fake = FakeRepresentation("dep1", dependencies=[])
    repr1 = FakeRepresentation("repr1", dependencies=[dep1_fake])
    dep1 = FakeRepresentation("dep1", dependencies=[])
    video = VREVideo(np.random.randint(0, 255, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    with pytest.raises(ValueError):
        VRE(video, [repr1, dep1])

def test_vre_hidden_duplicates_2():
    """
    this test looks for "hidden" dupplicates: i.e. the dependencies of a representation differs from the main one
    For example, in this case:
    - repr1 <- dep1
    - repr2 <- dep2 <- dep1 (fake)

    We don't explicitly check that dep1 (main) and dep1 (dep) are the same object (or at least the same type/params)
    """

    dep1_fake = FakeRepresentation("dep1", dependencies=[])
    dep1 = FakeRepresentation("dep1", dependencies=[])
    dep2 = FakeRepresentation("dep2", dependencies=[dep1_fake])
    repr1 = FakeRepresentation("repr1", dependencies=[dep1])
    repr2 = FakeRepresentation("repr2", dependencies=[dep2])
    video = VREVideo(np.random.randint(0, 255, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    with pytest.raises(ValueError):
        VRE(video, [repr1, repr2, dep1, dep2])

if __name__ == "__main__":
    test_vre_hidden_duplicates_1()
