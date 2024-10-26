"""
Integration test that checks that we don't call vre_setup if not needed.
Particulary, we have 2 representations: r1 (learned) and r2 (dep of r1).
r1 has its data on disk already so it should only load without ever calling vre_setup() when we compute r2
"""
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from vre import VideoRepresentationsExtractor as VRE
from vre.representations import Representation, LearnedRepresentationMixin, ReprOut
from vre.utils import FakeVideo

class MyRepresentation(Representation, LearnedRepresentationMixin):
    def __init__(self, name, dependencies):
        super().__init__(name, dependencies)
        self.vre_setup_called = False
        self.vre_free_called = False
        self.make_called = False

    def make(self, frames, dep_data = None):
        self.make_called = True
        return ReprOut(frames)
    def make_images(self, frames, repr_data):
        return repr_data.output
    def size(self, repr_data):
        raise NotImplementedError
    def resize(self, repr_data, new_size):
        raise NotImplementedError
    def vre_setup(self, load_weights = True):
        self.vre_setup_called = True
    def vre_free(self):
        self.vre_free_called = True

class MyDependentRepresentation(Representation):
    def make(self, frames, dep_data = None):
        return ReprOut(dep_data["r1"].output)
    def make_images(self, frames, repr_data):
        return repr_data.output
    def size(self, repr_data):
        raise NotImplementedError
    def resize(self, repr_data, new_size):
        raise NotImplementedError

def test_no_vre_setup_if_not_needed():
    tmp_dir = Path(TemporaryDirectory().name)
    (tmp_dir / "r1/npz").mkdir(parents=True, exist_ok=False)
    video = FakeVideo(np.random.randint(0, 255, size=(10, 20, 20, 3)).astype(np.uint8), frame_rate=30)
    for i in range(len(video)):
        np.savez(f"{tmp_dir}/r1/npz/{i}.npz", video[i])
    r1 = MyRepresentation("r1", [])
    r2 = MyDependentRepresentation("r2", [r1])
    vre = VRE(video, {"r1": r1, "r2": r2})

    vre.run(tmp_dir, start_frame=0, end_frame=None, binary_format="npz", image_format=None,
            output_dir_exists_mode="skip_computed", load_from_disk_if_computed=True, output_size="native")
    assert r1.make_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
    # since all the data of r1 is already on the disk, there's no need to call vre_setup and vre_free at all
    assert r1.vre_setup_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
    assert r1.vre_free_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
