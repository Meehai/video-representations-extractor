import sys
from pathlib import Path
import pytest
import numpy as np

from vre_video import VREVideo
from vre.utils import get_project_root
from vre.vre_runtime_args import VRERuntimeArgs
from vre.run_metadata import RunMetadata
from vre.representation_metadata import RepresentationMetadata

sys.path.append(str(get_project_root() / "test/vre"))
from fake_representation import FakeRepresentation


def test_RunMetadata_two_representations(tmp_path: Path):
    r1, r2 = FakeRepresentation("r1", dependencies=[]), FakeRepresentation("r2", dependencies=[])
    video = VREVideo(np.random.randint(0, 255, size=(10, 20, 30, 3), dtype=np.uint8), fps=1)
    runtime_args = VRERuntimeArgs(video, [r1, r2], [0, 1, 2, 3, 4, 5], "stop_execution", 0)
    run_metadata = RunMetadata(["r1", "r2"], runtime_args, logs_dir=tmp_path)
    metadata_r1 = RepresentationMetadata("r1", tmp_path / "r2_metadata.json", [0, 1, 2, 3, 4, 5], ["fmt"])
    metadata_r2 = RepresentationMetadata("r2", tmp_path / "r1_metadata.json", [0, 1, 2, 3, 4, 5], ["fmt"])
    metadata_r1.add_time(0.1, [0, 1, 2], run_id=run_metadata.id)
    metadata_r2.add_time(0.1, [0, 1, 2], run_id=run_metadata.id)
    run_metadata.add_run_stats(metadata_r1)
    assert run_metadata.run_stats == {
        "r1": {"n_computed": 3, "n_failed": 0, "average_duration": 0.03},
    }
    # cannot run the same representation twice in the same run
    with pytest.raises(AssertionError):
        run_metadata.add_run_stats(metadata_r1)
    run_metadata.add_run_stats(metadata_r2)
    assert run_metadata.run_stats == {
        "r1": {"n_computed": 3, "n_failed": 0, "average_duration": 0.03},
        "r2": {"n_computed": 3, "n_failed": 0, "average_duration": 0.03}
    }

if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        test_RunMetadata_two_representations(Path(tmp))
