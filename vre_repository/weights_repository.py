"""Weights Repository module for Learnable Representations in the default repository"""
from pathlib import Path
from urllib.request import urlretrieve
import os

from vre.utils import get_project_root, DownloadProgressBar, is_git_lfs
from vre.logger import vre_logger as logger

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
RESOURCES_URL = f"{REPO_URL}/-/raw/master/resources"
RESOURCES_DIR = Path(os.getenv("VRE_RESOURCES_DIR", get_project_root() / "resources"))

def fetch_resource(resource_name: str) -> Path:
    """fetches a resources from gitlab LFS if needed"""
    url = f"{RESOURCES_URL}/{resource_name}"
    (path := RESOURCES_DIR / resource_name).parent.mkdir(exist_ok=True, parents=True)
    if not Path(path).exists() or is_git_lfs(path):
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=resource_name) as t:
            try:
                urlretrieve(url, filename=path, reporthook=t.update_to)
            except Exception as e:
                logger.info(f"Failed to download '{url}' to '{path}'")
                raise e
    return path

def fetch_weights(paths: list[str], depth: int = 0) -> list[Path]:
    """fetches weights for a representation from the repository (if needed) and returns the local path"""
    assert isinstance(paths, list), (paths, type(paths))
    assert depth <= 1, depth
    res = []
    for path in paths:
        if isinstance(path, list): # cases like marigold where 1 ckpt file is split across multiple ckpt files
            local_path = fetch_weights(path, depth + 1)[0].parent
        else:
            local_path = fetch_resource(f"weights/{path}")
        res.append(local_path)
    return res
