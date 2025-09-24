"""Weights Repository module for Learnable Representations in the default repository"""
from pathlib import Path
from urllib.request import urlretrieve

from vre.utils import DownloadProgressBar, is_git_lfs
from vre.logger import vre_logger as logger

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
VRE_REPO_URL = f"{REPO_URL}/-/raw/master/vre_repository"

def _fetch_from_repo(src: str, dst: Path) -> Path:
    if not dst.exists() or is_git_lfs(dst):
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dst.name) as t:
            try:
                urlretrieve(src, filename=dst, reporthook=t.update_to)
            except Exception as e:
                logger.info(f"Failed to download '{src}' to '{dst}'")
                raise e
    return dst

def fetch_weights(paths: list[str | Path]) -> list[Path]:
    """
    Fetches weights for a representation from the repository (if needed) and returns the local path.
    The paths must be of the format: "/path/to/vre_repository/..representation../weights/ckpt(s)".
    """
    assert isinstance(paths, list), (paths, type(paths))
    assert all(isinstance(p, (str, Path, list)) for p in paths), paths
    assert all(str(x).find("/vre_repository/") for x in paths), paths
    res = []
    for path in paths:
        src = VRE_REPO_URL + str(path)[str(path).index("vre_repository")+14:]
        if isinstance(path, list):
            res.append(fetch_weights(path))
        else:
            res.append(_fetch_from_repo(src, path))
    return res
