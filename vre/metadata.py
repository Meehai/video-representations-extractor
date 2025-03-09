"""Metadata module -- The metadata of a particular VRE run"""
from __future__ import annotations
import json
import os
from typing import Any
from pathlib import Path
from pprint import pformat
import numpy as np

from .logger import vre_logger as logger
from .utils import str_maxk, AtomicOpen, random_chars
from .vre_runtime_args import VRERuntimeArgs

class SummaryPrinter:
    def __init__(self, representations: list[str], runtime_args: VRERuntimeArgs):
        self.representations = representations
        self.repr_metadatas: dict[str, RepresentationMetadata | None] = {r: None for r in representations}
        self.runtime_args = runtime_args.to_dict()

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

    def pretty_format(self) -> str:
        """returns a pretty formatted string of the metadata"""
        vre_run_stats_np = np.array([list(x.values()) for x in self.run_stats.values()]).T.round(3)
        vre_run_stats_np[abs(vre_run_stats_np - float(1<<31)) < 1e-2] = float("nan")
        res = ""
        frames = self.runtime_args["frames"]
        chosen_frames = sorted(np.random.choice(frames, size=min(5, len(frames)), replace=False))
        res = f"{'Name':<20}" + "|" + "|".join([f"{str_maxk(k, 9):<9}" for k in map(str, chosen_frames)]) + "|Total"
        try:
            for i, vrepr in enumerate(self.representations):
                res += "\n" + f"{str_maxk(vrepr, 20):<20}"
                for chosen_frame in chosen_frames:
                    res += "|" + f"{vre_run_stats_np[:, i][frames.index(chosen_frame)]:<9}"
                res += "|" + f"{vre_run_stats_np[:, i].sum().round(2)}"
        except:
            breakpoint()
        return res

class RunMetadata:
    """Metadata of a run for multiple representations. Backed on the disk by a JSON file"""
    def __init__(self, representations: list[str], runtime_args: VRERuntimeArgs, disk_location: Path):
        assert len(representations) > 0 and all(isinstance(x, str) for x in representations), representations
        self.representations = representations
        self.disk_location = disk_location
        self.runtime_args = runtime_args.to_dict()
        self.id = random_chars(n=10)
        self.data_writers = {}

    @property
    def metadata(self) -> dict[str, Any]:
        """the run metadata as a json"""
        return {
            "id": self.id,
            "runtime_args": self.runtime_args,
            "data_writers": self.data_writers,
        }

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)

    def __repr__(self):
        return (f"[Metadata] Id: {self.id}, Num representations: {len(self.representations)}. "
                f"Disk location: '{self.disk_location}'")

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
            if self.data_writer_meta.get("output_dir_exists_mode", "") == "overwrite":
                os.remove(disk_location)
        self.store_on_disk()

    @property
    def frames_computed(self) -> list[str]:
        """returns the list of comptued frames so far"""
        return [k for k, v in self.run_stats.items() if v is not None and v != 1<<31] # TODO: test

    def add_time(self, duration: float, frames: list[int]):
        """adds a (batched) time to the representation's run_stats"""
        assert (batch_size := len(frames)) > 0, batch_size
        data = [duration / batch_size] * batch_size
        for k, v in zip(map(str, frames), data): # make the frames strings due to json keys issue when storing/loading
            if self.run_stats[k] is not None and self.run_stats[k] != 1<<31:
                raise ValueError(f"Adding time to existing metadata {self}. Frame={k}. Previous: {self.run_stats[k]}")
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
                    loaded_v = loaded_json_data["run_stats"][k]
                    if loaded_v is not None and loaded_v != 1<<31:
                        assert v == loaded_v or v is None, (k, v, loaded_v)
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
