"""Metadata module -- The metadata of a particular VRE run"""
from __future__ import annotations
import json
from typing import Any
from pathlib import Path

from .representation_metadata import RepresentationMetadata
from .utils import random_chars, mean, now_fmt
from .vre_runtime_args import VRERuntimeArgs

class RunMetadata:
    """Metadata of a run for multiple representations. Backed on the disk by a JSON file"""
    def __init__(self, repr_names: list[str], runtime_args: VRERuntimeArgs, logs_dir: Path,
                 now_str: str | None = None, run_id: str | None = None):
        assert len(repr_names) > 0 and all(isinstance(x, str) for x in repr_names), repr_names
        assert (A := set(r.name for r in runtime_args.representations)) == (B := set(repr_names)), (A, B)
        self.repr_names = repr_names
        self.logs_dir = logs_dir
        self.runtime_args = runtime_args.to_dict()
        self.id = run_id or random_chars(n=10)
        self.data_writers = {}
        self.run_stats = {}
        self.total_stats = {
            "n_computed": 0,
            "n_failed": 0,
            "duration": 0,
        }
        self.now_str = now_str or now_fmt()
        self.store_on_disk()

    @property
    def disk_location(self) -> Path:
        """The full path disk location of this metadata"""
        return self.logs_dir / f"run_metadata-{self.id}-{self.now_str}.json"

    @property
    def metadata(self) -> dict[str, Any]:
        """the run metadata as a json"""
        return {
            "id": self.id,
            "runtime_args": self.runtime_args,
            "data_writers": self.data_writers,
            "run_stats": self.run_stats,
            "total": self.total_stats,
        }

    def add_run_stats(self, repr_metadata: RepresentationMetadata):
        """adds statistics of a single representation after it finished running"""
        assert (name := repr_metadata.repr_name) not in self.run_stats, f"{name} in {self.run_stats.keys()}"
        frames_computed = repr_metadata.frames_computed(run_id=self.id)
        frames_failed = repr_metadata.frames_failed(run_id=self.id)

        avg_duration = round(mean([repr_metadata.run_stats[ix].duration for ix in frames_computed]), 3)
        self.run_stats[name] = {"n_computed": len(frames_computed), "n_failed": len(frames_failed),
                                "average_duration": avg_duration}
        self.total_stats = {
            "n_computed": self.total_stats["n_computed"] + len(frames_computed),
            "n_failed": self.total_stats["n_failed"] + len(frames_failed),
            "duration": round(self.total_stats["duration"] + avg_duration * len(frames_computed), 3)
        }
        self.store_on_disk()

    def store_on_disk(self):
        """stores (overwrites if needed) the metadata on disk"""
        with open(self.disk_location, "w") as fp:
            json.dump(self.metadata, fp, indent=4)

    def __repr__(self):
        return (f"[Metadata] Id: {self.id}, Num representations: {len(self.repr_names)}. "
                f"Disk location: '{self.disk_location}'")
