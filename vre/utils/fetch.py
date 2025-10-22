"""fetch.py -- The VRE files fetcher"""
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

from ..logger import vre_logger as logger

class DownloadProgressBar(tqdm):
    """requests + tqdm"""
    def update_to(self, b=1, bsize=1, tsize=None):
        """Callback from tqdm"""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def is_git_lfs(path: Path) -> bool:
    """Returns true if a path is a git lfs link"""
    if path.is_dir():
        return False
    with open(path, "rb") as fp:
        try:
            return fp.read(7).decode("utf-8") == "version"
        except UnicodeDecodeError:
            return False

def fetch(src: str, dst: Path) -> Path:
    """fetches the source url to a destionat path. Returns that path. Support for git lfs and recursive mkdir()"""
    if not dst.exists() or is_git_lfs(dst):
        dst.parent.mkdir(exist_ok=True, parents=True)
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dst.name) as t:
            try:
                urlretrieve(src, filename=dst, reporthook=t.update_to)
            except Exception as e:
                logger.info(f"Failed to download '{src}' to '{dst}'")
                raise e
    return dst
