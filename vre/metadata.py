"""Metadata module -- The metadata of a particular VRE run"""
import json
from pathlib import Path

from .vre_runtime_args import VRERuntimeArgs

class Metadata:
    """Metadata of a run for multiple representations. Stored on the disk during the run too."""
    def __init__(self, representations: list[str], runtime_args: VRERuntimeArgs, disk_location: Path):
        assert len(representations) > 0 and all(isinstance(x, str) for x in representations), representations
        self.metadata = {"runtime_args": runtime_args.to_dict(), "data_writers": {}, "run_stats": {}}
        self.representations = representations
        self.disk_location = disk_location

    def add_time(self, representation: str, duration: float, batch_size: int):
        """adds a (batched) time to the representation's run_stats"""
        assert batch_size > 0, batch_size
        if representation not in self.metadata["run_stats"]:
            self.metadata["run_stats"][representation] = []
        data = [duration / batch_size] * batch_size if duration != 1 << 31 else [1 << 31] * batch_size
        self.metadata["run_stats"][representation].extend(data)

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)
