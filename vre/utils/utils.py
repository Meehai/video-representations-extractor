"""utils for vre"""
from typing import Any
from pathlib import Path
from datetime import datetime, timezone as tz
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

class DownloadProgressBar(tqdm):
    """requests + tqdm"""
    def update_to(self, b=1, bsize=1, tsize=None):
        """Callback from tqdm"""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class FixedSizeOrderedDict(OrderedDict):
    """An OrderedDict with a fixed size. Useful for caching purposes."""
    def __init__(self, *args, maxlen: int = 0, **kwargs):
        self._maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._maxlen > 0:
            if len(self) > self._maxlen:
                self.popitem(False)

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[2].absolute()

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]

def is_dir_empty(dir_path: Path, pattern: str = "*") -> bool:
    """returns true if directory is not empty, false if it is empty"""
    assert pattern.startswith("*"), pattern
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

def semantic_mapper(semantic_original: np.ndarray, mapping: dict[str, list[str]],
                    original_classes: list[str]) -> np.ndarray:
    """maps a bigger semantic segmentation to a smaller one"""
    assert len(semantic_original.shape) == 2, f"Only argmaxed data supported, got: {semantic_original.shape}"
    assert np.issubdtype(semantic_original.dtype, np.integer), semantic_original.dtype
    mapping_ix = {list(mapping.keys()).index(k): [original_classes.index(_v) for _v in v] for k, v in mapping.items()}
    flat_mapping = {}
    for k, v in mapping_ix.items():
        for _v in v:
            flat_mapping[_v] = k
    mapped_data = np.vectorize(flat_mapping.get)(semantic_original).astype(np.uint8)
    return mapped_data

def abs_path(x: str | Path) -> Path:
    """returns the absolute path of a string/path"""
    return Path(x).absolute()

def reorder_dict(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """simply puts in front the desired keys from the original dict, keeping the others intact"""
    assert (diff := set(keys).difference(data.keys())) == set(),diff
    for k in keys[::-1]:
        data = {k: data[k], **{k: v for k, v in data.items() if data != k}}
    return data

def str_maxk(s: str, k: int) -> str:
    """returns the string if it is smaller than k, otherwise returns the first k-2 and last 5"""
    return f"{s[0:k-7]}..{s[-5:]}" if len(s) > k else s
