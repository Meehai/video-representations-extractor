"""Metadata module -- The metadata of a particular VRE run"""
from __future__ import annotations
import json
import os
from pathlib import Path
from pprint import pformat
import numpy as np

from .utils import str_maxk, AtomicOpen
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
    def run_stats(self) -> dict[str, dict[str, float]]:
        """The run stats of each representation that was registered to this run or older ones (only computed values)"""
        res = {}
        for r in self.repr_metadatas.values():
            if r is not None:
                _res = {k: v for k, v in r.run_stats.items() if v is not None}
                if len(_res) > 0:
                    res[r.repr_name] = _res
        return res

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)

    def pretty_format(self) -> str:
        """returns a pretty formatted string of the metadata"""
        # make sure that each representation metadata computed only what we asked for
        assert [(set(x.keys()) == set(map(str, self.runtime_args["frames"]))) for x in self.run_stats.values()], \
            f"\n{self.run_stats}\n{self.runtime_args['frames']}"
        vre_run_stats_np = np.array([list(x.values()) for x in self.run_stats.values()]).T.round(3)
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
    Note: that this file may be updated from multiple processes running at the same time. For this reason, we have an
    extra layer protection based on json modification time via `os.path.getmtime` and we merge before storing to disk.
    """
    def __init__(self, repr_name: str, disk_location: Path, frames: list[int], data_writer_meta: dict | None = None):
        self.run_had_exceptions = False
        self.repr_name = repr_name
        self.disk_location = disk_location
        self.run_stats: dict[str, float | None] = {str(f): None for f in frames}
        self.data_writer_meta = data_writer_meta or {}
        if disk_location.exists():
            if data_writer_meta.get("output_dir_exists_mode", "") == "overwrite":
                os.remove(disk_location)
        self.store_on_disk()

    @property
    def frames_computed(self) -> list[str]:
        """returns the list of comptued frames so far"""
        return [v for v in self.run_stats.values() if v is not None]

    def add_time(self, duration: float, frames: list[int]):
        """adds a (batched) time to the representation's run_stats"""
        assert (batch_size := len(frames)) > 0, batch_size
        data = [duration / batch_size] * batch_size
        for k, v in zip(map(str, frames), data): # make the frames strings due to json keys issue when storing/loading
            assert self.run_stats[k] is None, (self.repr_name, self.disk_location, f"frame={k}", self.run_stats[k])
            self.run_stats[k] = v
        self.store_on_disk()

    def store_on_disk(self):
        """
        Stores (overwrites) the metadata on disk. Note: this does allows for multi processing on the same repr!
        Note: that disk_writer_meta will be overwritten by the last writer.
        """
        file_exists = self.disk_location.exists()
        with AtomicOpen(self.disk_location, "a+") as fp:
            if file_exists: # if it exists, we merge the disk data with existing data before overwriting
                fp.seek(0)
                loaded_json_data = json.loads(fp.read())
                assert (a := loaded_json_data["run_stats"].keys()) == (b := self.run_stats.keys()), f"\n- {a}\n- {b}"
                assert (a := loaded_json_data["name"]) == (b := self.repr_name), f"\n- {a}\n- {b}"
                for k, v in self.run_stats.items():
                    if loaded_json_data["run_stats"][k] is not None:
                        if v is not None: # make sure 2 processes didn't write the same frames.
                            assert loaded_json_data["run_stats"][k] == v, (k, v, loaded_json_data["run_stats"][k])
                        self.run_stats[k] = loaded_json_data["run_stats"][k]

            fp.seek(0)
            fp.truncate()
            json_data = {
                "name": self.repr_name,
                "run_stats": self.run_stats,
                "data_writer": self.data_writer_meta,
            }
            json.dump(json_data, fp, indent=4)

    def __repr__(self):
        return (f"[ReprMetadata] Representation: {self.repr_name}. Frames: {len(self.run_stats)} "
                f"(computed: {len(self.frames_computed)}). Disk location: '{self.disk_location}'")
