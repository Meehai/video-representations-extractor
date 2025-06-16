"""RepresentationMetadata -- the metadata of a representation stored on disk"""
from pathlib import Path
from typing import Any, NamedTuple
from io import FileIO
import json

from .utils import AtomicOpen

FrameStats = NamedTuple("frame_stats", duration=float, run_id=str, formats=list[str])

class RepresentationMetadata:
    """
    A class that defines the metadata and events that happened during a single representation's run.
    It is backed by a single sqlite file living in the representation's directory under .run_metadata.json.
    Note: that this file may be updated from multiple processes running at the same time. For this reason, we have an
    extra layer protection based on json modification time via `os.path.getmtime` and we merge before storing to disk.
    """
    def __init__(self, repr_name: str, disk_location: Path, frames: list[int], formats: list[str]):
        assert all(isinstance(x, int) for x in frames), frames
        assert len(formats) > 0, "formats cannot be empty, is your representation an IORepresentation?"
        self.repr_name = repr_name
        self.disk_location = disk_location
        self.run_had_exceptions = False
        self.formats = formats
        self.run_stats: dict[int, FrameStats | None] = {f: None for f in frames}
        self.store_on_disk()

    def frames_computed(self, run_id: str | None = None) -> list[int]:
        """returns the list of comptued frames so far for this representation"""
        def condition(frame_stats: FrameStats | None) -> bool:
            if frame_stats is None or frame_stats.duration is None:
                return False
            return set(self.formats).issubset(frame_stats.formats)
        return [ix for ix, v in self.run_stats.items() if condition(v)
                and (v.run_id == run_id if run_id is not None else True)]

    def frames_failed(self, run_id: str | None = None) -> list[int]:
        """returns the list of failed frames so far for this representation"""
        return [ix for ix, v in self.run_stats.items() if v is not None and v.duration is None
                and (v.run_id == run_id if run_id is not None else True)]

    def add_time(self, duration: float | None, frames: list[int], run_id: str, sync: bool=False):
        """adds a (batched) time to the representation's run_stats. If sync is true, it also calls store_on_disk."""
        assert (batch_size := len(frames)) > 0, batch_size
        data = [duration / batch_size] * batch_size if duration is not None else [None] * batch_size
        for ix, frame_duration in zip(frames, data):
            if self.run_stats[ix] is not None and self.run_stats[ix].duration is not None:
                if set(self.formats).issubset(self.run_stats[ix].formats):
                    raise ValueError(f"Adding time to existing metadata {self}. Frame={ix}. "
                                     f"Previous: {self.run_stats[ix]}")
                # overwrite the run id but add the previous time of the old representations to current frame duration
                frame_duration += self.run_stats[ix].duration
            self.run_stats[ix] = FrameStats(run_id=run_id, duration=frame_duration, formats=self.formats)
        if sync:
            self.store_on_disk()

    def to_dict(self, sync: bool=False) -> dict:
        """returns the metadata as a dict. Used mostly for tests. If atomic is set to true, uses AtomicOpen"""
        if sync and self.disk_location.exists():
            with AtomicOpen(self.disk_location, "a+") as fp:
                self.run_stats = self._load_and_merge_with_metadata_from_disk(fp)
                fp.seek(0)
                fp.truncate()

        return {
            "name": self.repr_name,
            "run_stats": {k: None if v is None else v._asdict() for k, v in self.run_stats.items()},
        }

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
                "run_stats": {k: None if v is None else v._asdict() for k, v in self.run_stats.items()},
            }
            json.dump(json_data, fp, indent=4)

    def _load_and_merge_with_metadata_from_disk(self, fp: FileIO) -> dict[str, Any]:
        """given an (mutexed) fp, read the metadata from the disk and return the run stats. Fail if any errors"""
        fp.seek(0)
        raw_data = fp.read()
        loaded_json_data = json.loads(raw_data)
        loaded_run_stats = {int(k): None if v is None else FrameStats(**v)
                            for k, v in loaded_json_data["run_stats"].items()}
        assert (a := loaded_run_stats.keys()) == (b := self.run_stats.keys()), f"\n- {a}\n- {b}"
        assert (a := loaded_json_data["name"]) == (b := self.repr_name), f"\n- {a}\n- {b}"
        merged_run_stats = {}
        # TODO... test this stuff below
        for k, v in self.run_stats.items():
            if (loaded_v := loaded_run_stats[k]) is not None and loaded_v.duration is not None:
                if v is None:
                    merged_run_stats[k] = loaded_v
                    continue
                # v exists too below
                if set(self.formats) == set(loaded_v.formats) and v != loaded_v:
                    raise ValueError(k, v, loaded_v)
                # v exists and is different from loaded_v (i.e. jpg in new run and npz in prev) -> merge formats
                merged_formats = sorted(set(v.formats).union(loaded_v.formats))
                merged_run_stats[k] = FrameStats(duration=v.duration, formats=merged_formats, run_id=v.run_id)
            else:
                merged_run_stats[k] = v
        return merged_run_stats

    def __repr__(self):
        return (f"[ReprMetadata] Representation: {self.repr_name}. Frames: {len(self.run_stats)} "
                f"(computed: {len(self.frames_computed())}, failed: {len(self.frames_failed())}) "
                f"Disk location: '{self.disk_location}'")
