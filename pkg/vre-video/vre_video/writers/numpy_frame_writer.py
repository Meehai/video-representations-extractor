"""numpy_frame_writer - Module that implements frame writing using numpy"""
from pathlib import Path
import os
from tqdm import trange
import numpy as np

from .frame_writer import FrameWriter

class NumpyFrameWriter(FrameWriter):
    """"NumpyFrameWriter implementation. Writes videos to disk using numpy."""
    def __init__(self, fmt: str = "npy", compress: bool = False):
        super().__init__()
        assert fmt in {"npy", "npz"}, fmt
        self.format = fmt
        assert compress is False or fmt == "npz", "compress cannot be set for npy"
        self.compress = compress

    def write(self, video: "VREVideo", out_path: str | Path, start_frame: int = 0, end_frame: int | None = None):
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        assert len(files := list(out_path.iterdir())) == 0, f"Data exists in out_path. N files: {len(files)}"
        start_frame = start_frame or 0
        end_frame = end_frame or len(video)
        assert start_frame >= 0 and end_frame <= len(video), (start_frame, end_frame)

        write_fn = np.save
        if self.format == "npz":
            write_fn = np.savez_compressed if self.compress else np.savez

        for i in trange(start_frame, end_frame, disable=os.getenv("VRE_VIDEO_PBAR", "1") == "0",
                        desc=f"[NumpyFrameWriter] {out_path}"):
            write_fn(f"{out_path}/{i}", video[i])
