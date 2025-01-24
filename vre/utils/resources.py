"""Weights utils. Stuff related to weights repository, downloading and loading weights"""
import urllib.request
from pathlib import Path

from .utils import get_project_root, is_git_lfs, DownloadProgressBar

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
RESOURCES_URL = f"{REPO_URL}/-/raw/master/resources"
RESOURCES_DIR = get_project_root() / "resources"

def fetch_resource(resource_name: str) -> Path:
    """fetches a resources from gitlab LFS if needed"""
    url = f"{RESOURCES_URL}/{resource_name}"
    path = get_project_root() / "resources" / resource_name
    if not Path(resource_name).exists() is is_git_lfs(path):
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=resource_name) as t:
            urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
    return path
