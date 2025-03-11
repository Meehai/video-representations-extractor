from pathlib import Path
import sys
import json
import pytest
from threading import Thread
import time
import numpy as np

from vre import FakeVideo
from vre.utils import get_project_root
from vre.metadata import RepresentationMetadata, RunMetadata
from vre.vre_runtime_args import VRERuntimeArgs
sys.path.append(str(get_project_root() / "test"))
from fake_representation import FakeRepresentation

def test_RepresentationMetadata_ctor(tmp_path: Path):
    metadata = RepresentationMetadata("repr_metadata", tmp_path/"metadata.json", [0, 1, 2, 3, 4, 5])
    loaded_json = json.load(open(metadata.disk_location, "r"))
    expected_json = {
        "name": "repr_metadata",
        "run_stats": {
            "0": None,
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None
        },
    }
    assert loaded_json == expected_json

def test_RepresentationMetadata_add_time_1(tmp_path: Path):
    metadata = RepresentationMetadata("repr_metadata", tmp_path/"metadata.json", [0, 1, 2, 3, 4, 5])
    metadata.add_time(0.1, [0, 1, 2])
    loaded_json = json.load(open(metadata.disk_location, "r"))
    assert all(loaded_json["run_stats"][i] == 0.03333333333333333 for i in ["0", "1", "2"])
    assert all(loaded_json["run_stats"][i] is None for i in ["3", "4", "5"])

    with pytest.raises(ValueError): # cannot add the same time twice
        metadata.add_time(1, [1])
    with pytest.raises(ValueError): # even if just some frames are overlapping
        metadata.add_time(1, [2, 3, 4])
    metadata.add_time(1, [3, 4])
    loaded_json = json.load(open(metadata.disk_location, "r"))
    assert all(loaded_json["run_stats"][i] == 0.03333333333333333 for i in ["0", "1", "2"])
    assert all(loaded_json["run_stats"][i] == 0.5 for i in ["3", "4"])
    assert loaded_json["run_stats"]["5"] is None

# note: can be flaky due to how we lock the files. Works 99% of the time.
@pytest.mark.flaky(reruns=3)
def test_RepresentationMetadata_add_time_in_threads(tmp_path: Path):
    def worker_fn(thread_ix: int, n_threads: int, tmp_path: Path):
        metadata = RepresentationMetadata("repr_metadata", tmp_path / "metadata.json", N := range(100))
        for i in N:
            if i % n_threads == thread_ix:
                metadata.add_time(thread_ix, [i])
            time.sleep(0.001)
    n_threads = 4
    threads: list[Thread] = []
    for i in range(n_threads):
        threads.append(thr := Thread(target=worker_fn, args=(i, n_threads, tmp_path)))
        thr.start()
    [thr.join() for thr in threads]

    loaded_json = json.load(open(tmp_path / "metadata.json", "r"))
    for i in range(100):
        assert loaded_json["run_stats"][str(i)] == i % n_threads

def test_RunMetadata_two_representations(tmp_path: Path):
    r1, r2 = FakeRepresentation("r1", dependencies=[]), FakeRepresentation("r2", dependencies=[])
    video = FakeVideo(np.random.randint(0, 255, size=(10, 20, 30, 3)), fps=1)
    runtime_args = VRERuntimeArgs(video, [r1, r2], [0, 1, 2, 3, 4, 5], "stop_execution", 0)
    run_metadata = RunMetadata(["r1", "r2"], runtime_args, disk_location=tmp_path / "run_metadata.json")
    metadata_r1 = RepresentationMetadata("r1", tmp_path/"r2_metadata.json", [0, 1, 2, 3, 4, 5])
    metadata_r2 = RepresentationMetadata("r2", tmp_path/"r1_metadata.json", [0, 1, 2, 3, 4, 5])
    metadata_r1.add_time(0.1, [0, 1, 2])
    metadata_r2.add_time(0.1, [0, 1, 2])
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
