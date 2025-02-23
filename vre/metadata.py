"""Metadata module -- The metadata of a particular VRE run"""
from __future__ import annotations
import json
from pathlib import Path
from pprint import pformat
import numpy as np

from .utils import str_maxk
from .vre_runtime_args import VRERuntimeArgs

class RunMetadata:
    """Metadata of a run for multiple representations. Backed on the disk by a JSON file"""
    def __init__(self, representations: list[str], runtime_args: VRERuntimeArgs, disk_location: Path):
        assert len(representations) > 0 and all(isinstance(x, str) for x in representations), representations
        self.metadata = {"runtime_args": runtime_args.to_dict()}
        self.representations = representations
        self.disk_location = disk_location
        self.repr_metadatas: dict[str, RepresentationMetadata | None] = {r: None for r in representations}

    @property
    def runtime_args(self) -> dict:
        """returns the runtime_args of the metadata"""
        return self.metadata["runtime_args"]

    @property
    def run_stats(self) -> dict[str, list[float | None]]:
        """The run stats of each representation that was registered to this run"""
        return {r: list(r.run_stats.values()) for r in self.repr_metadatas.values() if r is not None}

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)

    def pretty_format(self) -> str:
        """returns a pretty formatted string of the metadata"""
        vre_run_stats_np = np.array([list(meta.run_stats.values()) for meta in self.repr_metadatas.values()]).T.round(3)
        vre_run_stats_np[abs(vre_run_stats_np - float(1<<31)) < 1e-2] = float("nan")
        res = f"{'ix':<5}" + "|" + "|".join([f"{str_maxk(k, 20):<20}" for k in self.representations])
        frames = self.runtime_args["frames"]
        for frame in sorted(np.random.choice(frames, size=min(5, len(frames)), replace=False)):
            ix = frames.index(frame)
            res += "\n" + f"{frame:<5}" + "|" + "|".join([f"{str_maxk(str(v), 20):<20}" for v in vre_run_stats_np[ix]])
        res += "\n" + f"\nTotal:\n{pformat(dict(zip(self.representations, vre_run_stats_np.sum(0).round(3))))}"
        return res

    def __repr__(self):
        return f"[Metadata] Num representations: {len(self.representations)}. Disk location: '{self.disk_location}'"

class RepresentationMetadata:
    """
    A class that defines the metadata and events that happened during a single representation's run.
    It is backed by a single sqlite file living in the representation's directory under .run_metadata.json.
    """
    def __init__(self, repr_name: str, disk_location: Path, frames: list[int], data_writer_meta: dict | None):
        self.run_had_exceptions = False
        self.repr_name = repr_name
        self.disk_location = disk_location
        self.run_stats: dict[str, float | None] = {str(f): None for f in frames}
        self.data_writer_meta = data_writer_meta or {}

    def add_time(self, duration: float, frames: list[int]):
        """adds a (batched) time to the representation's run_stats"""
        assert (batch_size := len(frames)) > 0, batch_size
        data = [duration / batch_size] * batch_size if duration != 1 << 31 else [1 << 31] * batch_size
        for k, v in zip(map(str, frames), data): # make the frames strings due to json keys issue when storing/loading
            assert self.run_stats[k] is None, (self.repr_name, self.disk_location, f"frame={k}", self.run_stats[k])
            self.run_stats[k] = v

    def store_on_disk(self):
        """stores (overwrites) the metadata on disk. Note: this does not allow for multi processing on the same repr!"""
        json_data = {
            "name": self.repr_name,
            "run_stats": self.run_stats,
            "data_writer": self.data_writer_meta,
        }
        with open(self.disk_location, "w") as fp:
            json.dump(json_data, fp, indent=4)

    @staticmethod
    def build_from_disk(path: Path) -> RepresentationMetadata:
        """builds the representation metadata from disk"""
        with open(path, "r") as fp:
            data = json.load(fp)
        assert path.parent.name == data["name"], f"disk: {path.parent.name} vs. loaded: {data['name']}"
        res = RepresentationMetadata(data["name"], path, len(data["run_stats"]), data["data_writer"])
        res.run_stats = data["run_stats"]
        return res

    def __repr__(self):
        return (f"[ReprMetadata] Representation: {self.repr_name}. Frames: {len(self.run_stats)} "
                f"(comptuted: {sum(v is not None for v in self.run_stats)}). Disk location: '{self.disk_location}'")
