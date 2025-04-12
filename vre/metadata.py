"""Metadata module -- The metadata of a particular VRE run"""
from __future__ import annotations
import json
from io import FileIO
from typing import Any
from pathlib import Path

from .utils import AtomicOpen, random_chars, mean
from .vre_runtime_args import VRERuntimeArgs

class RunMetadata:
    """Metadata of a run for multiple representations. Backed on the disk by a JSON file"""
    def __init__(self, repr_names: list[str], runtime_args: VRERuntimeArgs, disk_location: Path):
        assert len(repr_names) > 0 and all(isinstance(x, str) for x in repr_names), repr_names
        assert (A := set(r.name for r in runtime_args.representations)) == (B := set(repr_names)), (A, B)
        self.repr_names = repr_names
        self.disk_location = disk_location
        self.runtime_args = runtime_args.to_dict()
        self.id = random_chars(n=10)
        self.data_writers = {}
        self.run_stats = {}
        self.store_on_disk()

    @property
    def metadata(self) -> dict[str, Any]:
        """the run metadata as a json"""
        return {
            "id": self.id,
            "runtime_args": self.runtime_args,
            "data_writers": self.data_writers,
            "run_stats": self.run_stats,
        }

    def add_run_stats(self, repr_metadata: RepresentationMetadata):
        """adds statistics of a single representation after it finished running"""
        assert (name := repr_metadata.repr_name) not in self.run_stats, f"{name} in {self.run_stats.keys()}"
        frames_computed = repr_metadata.frames_computed(this_run_only=True)
        frames_failed = repr_metadata.frames_failed(this_run_only=True)

        avg_duration = round(mean([repr_metadata.run_stats[ix]["duration"] for ix in frames_computed]), 2)
        self.run_stats[name] = {"n_computed": len(frames_computed), "n_failed": len(frames_failed),
                                "average_duration": avg_duration}
        self.store_on_disk()

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)

    def __repr__(self):
        return (f"[Metadata] Id: {self.id}, Num representations: {len(self.repr_names)}. "
                f"Disk location: '{self.disk_location}'")

class RepresentationMetadata:
    """
    A class that defines the metadata and events that happened during a single representation's run.
    It is backed by a single sqlite file living in the representation's directory under .run_metadata.json.
    Note: that this file may be updated from multiple processes running at the same time. For this reason, we have an
    extra layer protection based on json modification time via `os.path.getmtime` and we merge before storing to disk.
    """
    def __init__(self, repr_name: str, run_id: str, disk_location: Path, frames: list[int]):
        assert all(isinstance(x, int) for x in frames), frames
        self.run_id = run_id
        self.repr_name = repr_name
        self.disk_location = disk_location
        self.run_had_exceptions = False
        self.run_stats: dict[int, float | None] = {f: None for f in frames}
        self.store_on_disk()

    def frames_computed(self, this_run_only: bool=False) -> list[int]:
        """returns the list of comptued frames so far for this representation"""
        return [ix for ix, v in self.run_stats.items() if v is not None and v["duration"] is not None
                and (v["run_id"] == self.run_id if this_run_only else True)]

    def frames_failed(self, this_run_only: bool=False) -> list[int]:
        """returns the list of failed frames so far for this representation"""
        return [ix for ix, v in self.run_stats.items() if v is not None and v["duration"] is None
                and (v["run_id"] == self.run_id if this_run_only else True)]

    def add_time(self, duration: float | None, frames: list[int]):
        """adds a (batched) time to the representation's run_stats"""
        assert (batch_size := len(frames)) > 0, batch_size
        data = [duration / batch_size] * batch_size if duration is not None else [None] * batch_size
        for ix, frame_duration in zip(frames, data):
            if self.run_stats[ix] is not None and self.run_stats[ix]["duration"] is not None:
                raise ValueError(f"Adding time to existing metadata {self}. Frame={ix}. Previous: {self.run_stats[ix]}")
            self.run_stats[ix] = {"run_id": self.run_id, "duration": frame_duration}
        self.store_on_disk()

    def store_on_disk(self):
        """
        Stores (overwrites) the metadata on disk. Note: this does allows for multi processing on the same repr!
        Note: that disk_writer_meta will be overwritten by the last writer.
        """
        file_exists = self.disk_location.exists()
        with AtomicOpen(self.disk_location, "a+") as fp:
            if file_exists: # if it exists, we merge the disk data with existing data before overwriting
                self.run_stats = self._load_and_merge_with_metadata_from_disk(fp)

            fp.seek(0)
            fp.truncate()
            json_data = {
                "name": self.repr_name,
                "run_stats": self.run_stats,
            }
            json.dump(json_data, fp, indent=4)

    def _load_and_merge_with_metadata_from_disk(self, fp: FileIO) -> dict[str, Any]:
        """given an (mutexed) fp, read the metadata from the disk and return the run stats. Fail if any errors"""
        fp.seek(0)
        loaded_json_data = json.loads(fp.read())
        loaded_run_stats = {int(k): v for k, v in loaded_json_data["run_stats"].items()} # convert from json keys to int
        assert (a := loaded_run_stats.keys()) == (b := self.run_stats.keys()), f"\n- {a}\n- {b}"
        assert (a := loaded_json_data["name"]) == (b := self.repr_name), f"\n- {a}\n- {b}"
        merged_run_stats = {}
        for k, v in self.run_stats.items():
            if (loaded_v := loaded_run_stats[k]) is not None and loaded_v["duration"] is not None:
                assert v == loaded_v or v is None, (k, v, loaded_v)
                merged_run_stats[k] = loaded_v
            else:
                merged_run_stats[k] = v
        return merged_run_stats

    def __repr__(self):
        return (f"[ReprMetadata] Representation: {self.repr_name}. Frames: {len(self.run_stats)} "
                f"(computed: {len(self.frames_computed())}, failed: {len(self.frames_failed())}) "
                f"Disk location: '{self.disk_location}'")
