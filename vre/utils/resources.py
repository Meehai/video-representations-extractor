"""Weights utils. Stuff related to weights repository, downloading and loading weights"""
from pathlib import Path
import urllib.request
import os

from .utils import get_project_root, DownloadProgressBar

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
RESOURCES_URL = f"{REPO_URL}/-/raw/master/resources"
RESOURCES_DIR = Path(os.getenv("VRE_RESOURCES_DIR", get_project_root() / "resources"))

def fetch_resource(resource_name: str) -> Path:
    """fetches a resources from gitlab LFS if needed"""
    url = f"{RESOURCES_URL}/{resource_name}"
    path = RESOURCES_DIR / resource_name
    if not Path(resource_name).exists():
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=resource_name) as t:
            urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path
