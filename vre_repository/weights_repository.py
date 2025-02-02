"""Weights Repository module for Learnable Representations in the default repository"""
from pathlib import Path
from vre.utils.utils import is_git_lfs
from vre.utils.resources import RESOURCES_DIR, fetch_resource

def fetch_weights(paths: list[str], depth: int = 0) -> list[Path]:
    """fetches weights for a representation from the repository (if needed) and returns the local path"""
    assert isinstance(paths, list), (paths, type(paths))
    assert depth <= 1, depth
    res = []
    for path in paths:
        if isinstance(path, list): # cases like marigold where 1 ckpt file is split across multiple ckpt files
            local_path = fetch_weights(path, depth + 1)[0].parent
        else:
            local_path = RESOURCES_DIR / "weights" / path
        res.append(local_path)
        if local_path.exists() and not is_git_lfs(local_path): # is_git_lfs is for CI where weights are lfs links
            continue
        fetch_resource(f"weights/{path}")
    return res
