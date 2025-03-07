from vre.metadata import RepresentationMetadata
from pathlib import Path
import json
import pytest

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
        "data_writer": {}
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
