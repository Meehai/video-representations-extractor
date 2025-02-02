"""Weights utils. Stuff related to weights repository, downloading and loading weights"""
from pathlib import Path
from urllib.request import urlretrieve
import os

from .utils import get_project_root, DownloadProgressBar, is_git_lfs
from ..logger import vre_logger as logger

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
