"""Weights Repository module for Learnable Representations in the default repository"""
from pathlib import Path

from vre.utils import fetch

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
VRE_REPO_URL = f"{REPO_URL}/-/raw/master/vre_repository"

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
            res.append(fetch(src, path))
    return res
