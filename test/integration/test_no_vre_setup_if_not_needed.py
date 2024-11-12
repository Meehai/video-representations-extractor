"""
Integration test that checks that we don't call vre_setup if not needed.
Particulary, we have 2 representations: r1 (learned) and r2 (dep of r1).
r1 has its data on disk already so it should only load without ever calling vre_setup() when we compute r2
"""
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from vre import VideoRepresentationsExtractor as VRE
from vre.representations import (
    Representation, LearnedRepresentationMixin, ReprOut, ComputeRepresentationMixin, NpIORepresentation)
from vre.utils import FakeVideo

class MyRepresentation(Representation, LearnedRepresentationMixin, ComputeRepresentationMixin, NpIORepresentation):
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        LearnedRepresentationMixin.__init__(self)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
        self.vre_setup_called = False
        self.vre_free_called = False
        self.make_called = False

    def compute(self, video, ixs):
        self.make_called = True
        self.data = ReprOut(frames := np.array(video[ixs]), output=frames, key=ixs)
    def make_images(self):
        return self.data.output
    def vre_setup(self, load_weights = True):
        self.vre_setup_called = True
    def vre_free(self):
        self.vre_free_called = True
        self.setup_called = False

class MyDependentRepresentation(Representation, ComputeRepresentationMixin, NpIORepresentation):
    def __init__(self, *args, **kwargs):
        Representation.__init__(self, *args, **kwargs)
        ComputeRepresentationMixin.__init__(self)
        NpIORepresentation.__init__(self)
    def compute(self, video, ixs):
        self.data = ReprOut(frames=np.array(video[ixs]), output=self.dependencies[0].data.output, key=ixs)
    def make_images(self):
        return self.data.output

def test_no_vre_setup_if_not_needed():
    tmp_dir = Path(TemporaryDirectory().name)
    (tmp_dir / "r1/npz").mkdir(parents=True, exist_ok=False)
    video = FakeVideo(np.random.randint(0, 255, size=(10, 20, 20, 3)).astype(np.uint8), fps=30)
    for i in range(len(video)):
        np.savez(f"{tmp_dir}/r1/npz/{i}.npz", video[i])
    r1 = MyRepresentation("r1", [])
    r2 = MyDependentRepresentation("r2", [r1])
    vre = VRE(video, {"r1": r1, "r2": r2}) \
        .set_compute_params(output_size="native", output_dtype="uint8") \
        .set_io_parameters(binary_format="npz")
    vre.run(tmp_dir, frames=list(range(0, len(video))), output_dir_exists_mode="skip_computed")
    assert r1.make_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
    # since all the data of r1 is already on the disk, there's no need to call vre_setup and vre_free at all
    assert r1.vre_setup_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
    assert r1.vre_free_called is False, (r1.make_called, r1.vre_setup_called, r1.vre_free_called)
