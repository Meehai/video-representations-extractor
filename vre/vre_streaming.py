"""vre_streaming -- module that implements vre[frames] so it goes frame by frame, not repr by repr"""
from tempfile import TemporaryDirectory
from pathlib import Path
# from .video_representations_extractor import VideoRepresentationsExtractor as VRE # pylint: disable=cyclic-import
from .utils import ReprOut, random_chars
from .vre_runtime_args import VRERuntimeArgs
from .representations.io_representation_mixin import ImageFormat

class VREStreaming:
    """VREStreaming class that implements vre[frames] so it goes frame by frame, not repr by repr"""
    def __init__(self, vre: "VRE", output_dir: Path | None = None, run_id: str | None = None):
        self.output_dir = output_dir or Path(TemporaryDirectory().name)
        self.vre = vre
        self.run_id: str | None = run_id or random_chars(n=10)

    def __getitem__(self, ix: int | slice | list[int]) -> dict[str, ReprOut]:
        if isinstance(ix, int):
            return self.__getitem__([ix])
        if isinstance(ix, slice):
            return self.__getitem__(list(range(ix.start, ix.stop)))
        res: dict[str, ReprOut] = {}
        for vre_repr in self.vre.representations:
            (self.output_dir / vre_repr.name).mkdir(exist_ok=True, parents=True)
            runtime_args = VRERuntimeArgs(video=self.vre.video, representations=self.vre.representations, frames=ix,
                                          exception_mode="stop_execution", n_threads_data_storer=1)
            _ = self.vre.do_one_representation(run_id=self.run_id, representation=vre_repr, output_dir=self.output_dir,
                                          output_dir_exists_mode="skip_computed", runtime_args=runtime_args)
            res[vre_repr.name] = self.vre._load_from_disk_if_possible(vre_repr, self.vre.video,
                                                                      ixs=list(ix), output_dir=self.output_dir)
            if vre_repr.image_format != ImageFormat.NOT_SET:
                res[vre_repr.name].output_images = vre_repr.make_images(res[vre_repr.name])
        return res
