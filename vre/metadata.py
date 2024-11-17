"""Metadata module -- The metadata of a particular VRE run"""
import json
from pathlib import Path
from pprint import pformat
import numpy as np

from .utils import str_maxk
from .vre_runtime_args import VRERuntimeArgs

class Metadata:
    """Metadata of a run for multiple representations. Stored on the disk during the run too."""
    def __init__(self, representations: list[str], runtime_args: VRERuntimeArgs, disk_location: Path):
        assert len(representations) > 0 and all(isinstance(x, str) for x in representations), representations
        self.metadata = {"runtime_args": runtime_args.to_dict(), "data_writers": {}, "run_stats": {}}
        self.representations = representations
        self.disk_location = disk_location

    @property
    def run_stats(self) -> dict[str, list[float]]:
        """returns the run_stats of the metadata"""
        return self.metadata["run_stats"]

    @property
    def runtime_args(self) -> dict:
        """returns the runtime_args of the metadata"""
        return self.metadata["runtime_args"]

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

    def pretty_format(self) -> str:
        """returns a pretty formatted string of the metadata"""
        vre_run_stats_np = np.array(list(self.metadata["run_stats"].values())).T.round(3)
        res = f"{'ix':<5}" + "|" + "|".join([f"{str_maxk(k, 15):<15}" for k in self.metadata["run_stats"].keys()])
        for i, frame in enumerate(self.metadata["runtime_args"]["frames"][0:5]):
            res += "\n" + f"{frame:<5}" + "|" + "|".join([f"{str_maxk(str(v), 15):<15}" for v in vre_run_stats_np[i]])
        res += "\n" + f"\nTotal:\n{pformat(dict(zip(self.metadata['run_stats'], vre_run_stats_np.sum(0).round(2))))}"
        return res
