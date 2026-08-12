"""pil_frame_writer - Module that implements frame writing using Pillow"""
from pathlib import Path
import os
from tqdm import trange

from .frame_writer import FrameWriter
from ..utils import image_write

class PILFrameWriter(FrameWriter):
    """"PILFrameWriter implementation. Writes videos to disk using PIL."""
    def __init__(self, fmt: str = "png"):
        super().__init__()
        assert fmt in {"png", "jpg"}, fmt
        self.suffix = fmt

    def write(self, video: "VREVideo", out_path: str | Path, start_frame: int = 0, end_frame: int | None = None):
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        assert len(files := list(out_path.iterdir())) == 0, f"Data exists in out_path. N files: {len(files)}"
        start_frame = start_frame or 0
        end_frame = end_frame or len(video)
        assert start_frame >= 0 and end_frame <= len(video), (start_frame, end_frame)
        for i in trange(start_frame, end_frame, disable=os.getenv("VRE_VIDEO_PBAR", "1") == "0",
                        desc=f"[PILFrameWriter] {out_path}"):
            image_write(video[i], f"{out_path}/{i}.{self.suffix}")
