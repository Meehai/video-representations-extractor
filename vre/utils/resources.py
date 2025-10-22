"""Resources utils. Stuff related to the resources stored in the repo via git-lfs. Weights are not in resources!"""
from pathlib import Path
import os

from .utils import get_project_root
from .fetch import fetch

REPO_URL = "https://gitlab.com/video-representations-extractor/video-representations-extractor"
RESOURCES_URL = f"{REPO_URL}/-/raw/master/resources"
RESOURCES_DIR = Path(os.getenv("VRE_RESOURCES_DIR", get_project_root() / "resources"))

def fetch_resource(resource_name: str) -> Path:
    """fetches a resources from gitlab LFS if needed"""
    return fetch(src=f"{RESOURCES_URL}/{resource_name}", dst=RESOURCES_DIR / resource_name)
