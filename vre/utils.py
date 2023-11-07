"""utils for vre"""
from typing import T
from pathlib import Path
import gdown
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from .logger import logger

def get_project_root() -> Path:
    """gets the root of this project"""
    return Path(__file__).parents[1]

def topological_sort(dependency_graph: dict[T, list[T]]) -> list[T]:
    """
    Given a graph as a dict {Node : [Dependencies]}, returns a list [Node] ordered with a correct topological
        sort order. Applies Kahn's algorithm: https://ocw.cs.pub.ro/courses/pa/laboratoare/laborator-07
    """
    L, S = [], []

    # First step is to create a regular graph of {Node : [Children]}
    graph: dict[str, T] = {k: [] for k in dependency_graph.keys()}
    in_nodes_graph = {}
    for key in dependency_graph:
        for parent in dependency_graph[key]:
            assert parent in graph, f"Node '{parent}' is not in given graph: {graph.keys()}"
            graph[parent].append(key)
        # Transform the depGraph into a list of number of in-nodes
        in_nodes_graph[key] = len(dependency_graph[key])
        # Add nodes with no dependencies and start BFS from them
        if in_nodes_graph[key] == 0:
            S.append(key)

    while len(S) > 0:
        u = S.pop()
        L.append(u)

        for v in graph[u]:
            in_nodes_graph[v] -= 1
            if in_nodes_graph[v] == 0:
                S.insert(0, v)

    for key in in_nodes_graph.keys():
        assert in_nodes_graph[key] == 0, "Graph has cycles. Cannot do topological sort."
    return L

def image_resize(x: np.ndarray, height: int, width: int) -> np.ndarray:
    """resizes an image to the given height and width"""
    dtype_orig = x.dtype
    x = x.astype(np.float32) / 255 if dtype_orig == np.uint8 else x
    y = resize(x, (height, width))
    y = (y * 255).astype(dtype_orig) if dtype_orig == np.uint8 else y
    return y

def image_write(x: np.ndarray, path: Path):
    """writes an image to a bytes string"""
    assert x.dtype == np.uint8, x.dtype
    imsave(path, x)
    logger.debug2(f"Saved image to '{path}'")

def gdown_mkdir(uri: str, local_path: Path):
    """calls gdown but also makes the directory of the parent path just to be sure it exists"""
    logger.debug(f"Downloading '{uri}' to '{local_path}'")
    local_path.parent.mkdir(exist_ok=True, parents=True)
    gdown.download(uri, f"{local_path}")
