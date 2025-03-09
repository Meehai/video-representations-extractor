"""utils for vre"""
from typing import Any, T, Callable
from pathlib import Path
from datetime import datetime, timezone as tz
from collections import OrderedDict
from math import sqrt
import random
import sys
import importlib
from tqdm import tqdm
import numpy as np
import torch as tr
from vre.logger import vre_logger as logger

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

def is_git_lfs(path: Path) -> bool:
    """Returns true if a path is a git lfs link"""
    if path.is_dir():
        return False
    with open(path, "rb") as fp:
        try:
            return fp.read(7).decode("utf-8") == "version"
        except UnicodeDecodeError:
            return False

def get_closest_square(n: int) -> tuple[int, int]:
    """
    Given a stack of N images, find the closest square X>=N*N and return that.
    Note: There are only 2 rows possible between x^2 and (x+1)^2 because (x+1)^2 = x^2 + 2*x + 1, thus we can add two
    columns at most. If a 3rd column is needed, then closest lower bound is (x+1)^2 and we must use that.
    Example: 9: 3*3; 12 -> 3*3 -> 3*4 (3 rows). 65 -> 8*8 -> 8*9. 73 -> 8*8 -> 8*9 -> 9*9
    """
    x = int(sqrt(n))
    r, c = x, x
    c = c + 1 if c * r < n else c
    r = r + 1 if c * r < n else r
    assert (c + 1) * r > n and c * (r + 1) > n
    return r, c

def now_fmt() -> str:
    """Returns now() as a UTC isoformat string"""
    return datetime.now(tz=tz.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")

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
    assert (A := len(set(flat_mapping))) == (B := len(set(original_classes))), (A, B)
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

def array_blend(x: np.ndarray, y: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Blends two arrays of the same shape with an alpha: number of array"""
    alpha_arr = np.asarray(alpha)
    assert x.shape == y.shape, (x.shape, y.shape)
    assert np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating), (x.dtype, y.dtype)
    assert (alpha_arr >= 0).all() and (alpha_arr <= 1).all(), (alpha_arr.min(), alpha_arr.max())
    try:
        return (1 - alpha_arr) * x + alpha_arr * y # the actual blend :)
    except Exception as e:
        logger.info(f"Exception thrown: {e}.\nShapes: {x.shape=} {y.shape=} {alpha_arr.shape=}")
        raise e

def make_batches(frames: list[int], batch_size: int) -> list[int]:
    """return 1D array [start_frame, start_frame+bs, start_frame+2*bs... end_frame]"""
     # TODO test all the cases of this fn
    if batch_size > len(frames):
        logger.warning(f"batch size {batch_size} is larger than #frames to process {len(frames)}.")
        batch_size = len(frames)
    if len(frames) == 0:
        return []
    batches, n_batches = [], len(frames) // batch_size + (len(frames) % batch_size > 0)
    for i in range(n_batches):
        batches.append(frames[i * batch_size: (i + 1) * batch_size])
    return batches

def vre_load_weights(path: Path) -> dict[str, tr.Tensor]:
    """load weights from disk. weights can be sharded as well. Sometimes we store this due to git-lfs big files bug."""
    logger.debug(f"Loading weights from '{path}'")

    if path.is_dir():
        res = {}
        for item in path.iterdir():
            res = {**res, **tr.load(item, map_location="cpu")}
        return res
    return tr.load(path, map_location="cpu")

def clip(x: T, _min: T, _max: T) -> T:
    """clips a value between [min, max]"""
    return max(min(x, _max), _min)

def load_function_from_module(module_path: str | Path, function_name: str) -> Callable:
    """Usage: fn = load_function_from_module("/path/to/stuff.py", "function_name"); y = fn(args);"""
    module_name = str(module_path).split("/")[-1].replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def random_chars(n: int) -> str:
    """returns a string of n random characters"""
    valid_chars = [*range(ord('A'), ord('Z')+1), *range(ord('a'), ord('z')+1), *range(ord('0'), ord('9')+1)]
    return "".join(map(chr, [random.choice(valid_chars) for _ in range(n)]))
