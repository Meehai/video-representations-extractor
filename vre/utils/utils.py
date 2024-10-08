"""utils for vre"""
from typing import Any
from pathlib import Path
from datetime import datetime, timezone as tz
from dataclasses import dataclass
import urllib.request
from tqdm import tqdm
import numpy as np

from ..logger import vre_logger as logger
from .fake_video import VREVideo

@dataclass
class RepresentationOutput:
    """The output of representation.make()"""
    output: np.ndarray
    extra: list[dict] | None = None

class DownloadProgressBar(tqdm):
    """requests + tqdm"""
    def update_to(self, b=1, bsize=1, tsize=None):
        """Callback from tqdm"""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[2].absolute()

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]

def to_categorical(data: np.ndarray, num_classes: int = None) -> np.ndarray:
    """converts the data to categorical. If num classes is not provided, it is infered from the data"""
    data = np.array(data)
    assert data.dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)
    if num_classes is None:
        num_classes = data.max()
    y = np.eye(num_classes)[data.reshape(-1)].astype(np.uint8)
    # Some bugs for (1, 1) shapes return (1, ) instead of (1, NC)
    MB = data.shape[0]
    y = np.squeeze(y)
    if MB == 1:
        y = np.expand_dims(y, axis=0)
    y = y.reshape(*data.shape, num_classes)
    return y

def took(prev: datetime.date, l: int, r: int) -> list[float]:
    """how much it took between [prev:now()] gien a [l:r] batch"""
    return [(datetime.now() - prev).total_seconds() / (r - l)] * (r - l)

def make_batches(video: VREVideo, start_frame: int, end_frame: int, batch_size: int) -> np.ndarray:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
    if batch_size > end_frame - start_frame:
        logger.warning(f"batch size {batch_size} is larger than #frames to process [{start_frame}:{end_frame}].")
        batch_size = end_frame - start_frame
    last_one = min(end_frame, len(video))
    batches = np.arange(start_frame, last_one, batch_size)
    batches = np.array([*batches, last_one], dtype=np.int64) if batches[-1] != last_one else batches
    return batches

def all_batch_exists(npy_paths: list[Path], png_paths: list[Path], l: int, r: int,
                     export_npy: bool, export_png: bool) -> bool:
    """checks whether all batches exist or not"""
    for ix in range(l, r):
        if export_npy and not npy_paths[ix].exists():
            return False
        if export_png and not png_paths[ix].exists():
            return False
    logger.debug(f"Batch [{l}:{r}] skipped.")
    return True

def is_dir_empty(dir_path: Path, pattern: str = "*") -> bool:
    """returns true if directory is not empty, false if it is empty"""
    return len(list(dir_path.glob(pattern))) == 0

def get_closest_square(n: int) -> tuple[int, int]:
    """
    Given a stack of N images
    Find the closest square X>=N*N and then remove rows 1 by 1 until it still fits X

    Example: 9: 3*3; 12 -> 3*3 -> 3*4 (3 rows). 65 -> 8*8 -> 8*9. 73 -> 8*8 -> 8*9 -> 9*9
    """

    x = int(np.sqrt(n))
    r, c = x, x
    # There are only 2 rows possible between x^2 and (x+1)^2 because (x+1)^2 = x^2 + 2*x + 1, thus we can add 2 columns
    #  at most. If a 3rd column is needed, then closest lower bound is (x+1)^2 and we must use that.
    if c * r < n:
        c += 1
    if c * r < n:
        r += 1
    assert (c + 1) * r > n and c * (r + 1) > n
    return r, c

def now_fmt() -> str:
    """Returns now() as a UTC isoformat string"""
    return datetime.now(tz=tz.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")

def is_git_lfs(path: Path) -> bool:
    """Returns true if a path is a git lfs link"""
    with open(path, "rb") as fp:
        try:
            return fp.read(7).decode("utf-8") == "version"
        except UnicodeDecodeError:
            return False

def fetch_resource(resource_name: str) -> Path:
    """fetches a resources from gitlab LFS if needed"""
    base_url = "https://gitlab.com/meehai/video-representations-extractor/-/raw/master/resources/"
    url = f"{base_url}/{resource_name}"
    path = get_project_root() / "resources" / resource_name
    if not Path(resource_name).exists() is is_git_lfs(path):
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=resource_name) as t:
            urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path
