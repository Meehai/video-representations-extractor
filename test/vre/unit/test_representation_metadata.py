from pathlib import Path
import json
import pytest
from threading import Thread
import time

from vre.representation_metadata import RepresentationMetadata

def test_RepresentationMetadata_ctor(tmp_path: Path):
    metadata = RepresentationMetadata("repr_metadata", tmp_path / "metadata.json", [0, 1, 2, 3, 4, 5], ["fmt"])
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
    metadata = RepresentationMetadata("repr_metadata", tmp_path / "metadata.json", [0, 1, 2, 3, 4, 5], ["fmt"])
    metadata.add_time(0.1, [0, 1, 2], "run_id")
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
        }
    }
    # not synced yet!
    assert loaded_json == expected_json

    memory_dict = metadata.to_dict()
    expected_dict = {
        "name": "repr_metadata",
        "run_stats": {
            0: {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            1: {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            2: {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            3: None,
            4: None,
            5: None
        }
    }
    assert memory_dict == expected_dict

    metadata.store_on_disk()
    loaded_json = json.load(open(metadata.disk_location, "r"))
    expected_json = {
        "name": "repr_metadata",
        "run_stats": {
            "0": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "1": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "2": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "3": None,
            "4": None,
            "5": None
        }
    }
    assert loaded_json == expected_json

    with pytest.raises(ValueError): # cannot add the same time twice
        metadata.add_time(1, [1], "run_id2")
    with pytest.raises(ValueError): # even if just some frames are overlapping
        metadata.add_time(1, [2, 3, 4], "run_id2")
    metadata.add_time(1, [3, 4], "run_id2", sync=True) # stores on disk
    loaded_json = json.load(open(metadata.disk_location, "r"))
    expected_json = {
        "name": "repr_metadata",
        "run_stats": {
            "0": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "1": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "2": {"run_id": "run_id", "formats": ["fmt"], "duration": 0.03333333333333333},
            "3": {"run_id": "run_id2", "formats": ["fmt"], "duration": 0.5},
            "4": {"run_id": "run_id2", "formats": ["fmt"], "duration": 0.5},
            "5": None
        }
    }

# note: can be flaky due to how we lock the files. Works 99% of the time.
@pytest.mark.flaky(reruns=3)
def test_RepresentationMetadata_add_time_in_threads(tmp_path: Path):
    def worker_fn(thread_ix: int, n_threads: int, tmp_path: Path):
        metadata = RepresentationMetadata("repr_metadata", tmp_path / "metadata.json", N := range(100), ["fmt"])
        for i in N:
            if i % n_threads == thread_ix:
                metadata.add_time(duration=thread_ix, frames=[i], run_id=f"run_id_{thread_ix}", sync=True)
            time.sleep(0.001)
    n_threads = 4
    threads: list[Thread] = []
    for i in range(n_threads):
        threads.append(thr := Thread(target=worker_fn, args=(i, n_threads, tmp_path)))
        thr.start()
    [thr.join() for thr in threads]

    loaded_json = json.load(open(tmp_path / "metadata.json", "r"))
    expected_json = {
        "name": "repr_metadata",
        "run_stats": {
            "0": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "1": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "2": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "3": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "4": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "5": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "6": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "7": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "8": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "9": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "10": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "11": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "12": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "13": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "14": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "15": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "16": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "17": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "18": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "19": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "20": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "21": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "22": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "23": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "24": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "25": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "26": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "27": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "28": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "29": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "30": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "31": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "32": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "33": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "34": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "35": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "36": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "37": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "38": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "39": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "40": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "41": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "42": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "43": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "44": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "45": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "46": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "47": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "48": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "49": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "50": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "51": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "52": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "53": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "54": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "55": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "56": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "57": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "58": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "59": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "60": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "61": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "62": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "63": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "64": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "65": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "66": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "67": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "68": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "69": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "70": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "71": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "72": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "73": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "74": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "75": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "76": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "77": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "78": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "79": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "80": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "81": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "82": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "83": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "84": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "85": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "86": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "87": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "88": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "89": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "90": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "91": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "92": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "93": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "94": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "95": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"},
            "96": {"duration": 0.0, "formats": ["fmt"], "run_id": "run_id_0"},
            "97": {"duration": 1.0, "formats": ["fmt"], "run_id": "run_id_1"},
            "98": {"duration": 2.0, "formats": ["fmt"], "run_id": "run_id_2"},
            "99": {"duration": 3.0, "formats": ["fmt"], "run_id": "run_id_3"}
        }
    }
    assert loaded_json == expected_json
