"""Build representation module. Updates faster than README :)"""
from typing import Type
from omegaconf import DictConfig, OmegaConf
import pims
from ..representation import Representation
from ..logger import logger
from ..utils import topological_sort


def build_representation_type(type: str, method: str) -> Type[Representation]:
    """Gets the representation type from a type and a method (the two identifiers of a representation)"""
    objType = None

    if type == "default":
        if method == "rgb":
            from .rgb import RGB

            objType = RGB
        elif method == "hsv":
            from .hsv import HSV

            objType = HSV

    elif type == "soft-segmentation":
        if method == "python-halftone":
            from .soft_segmentation.python_halftone import Halftone

            objType = Halftone
        elif method == "kmeans":
            from .soft_segmentation.kmeans import KMeans

            objType = KMeans
        elif method == "generalized_boundaries":
            from .soft_segmentation.generalized_boundaries import GeneralizedBoundaries

            objType = GeneralizedBoundaries

    elif type == "edges":
        if method == "dexined":
            from .edges.dexined import DexiNed

            objType = DexiNed
        elif method == "canny":
            from .edges.canny import Canny

            objType = Canny

    elif type == "depth":
        if method == "dpt":
            from .depth.dpt import DepthDpt

            objType = DepthDpt
        elif method == "odo-flow":
            from .depth.odo_flow import DepthOdoFlow

            objType = DepthOdoFlow

    elif type == "optical-flow":
        if method == "rife":
            from .optical_flow.rife import FlowRife

            objType = FlowRife
        elif method == "raft":
            from .optical_flow.raft import FlowRaft

            objType = FlowRaft

    elif type == "semantic":
        if method == "safeuav":
            from .semantic.safeuav import SSegSafeUAV

            objType = SSegSafeUAV

    elif type == "normals":
        if method == "depth-svd":
            from .normals.depth_svd import DepthNormalsSVD

            objType = DepthNormalsSVD

    assert objType is not None, f"Unknown type: {type}, method: {method}"
    return objType


def build_representation_from_cfg(video: pims.Video, repr_cfg: dict, name: str,
                                  built_so_far: dict[str, Representation]) -> Representation:
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    assert "type" in repr_cfg and "method" in repr_cfg, f"Broken format: {repr_cfg.keys()}"
    repr_type, repr_method = repr_cfg["type"], repr_cfg["method"]
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])
    logger.info(f"Building '{repr_type}'/'{name}'")

    obj_type = build_representation_type(repr_type, repr_method)
    # this is here because omegaconf transforms [1, 2, 3, 4] in a ListConfig, not a simple list
    obj = obj_type(video=video, name=name, dependencies=dependencies, **repr_cfg["parameters"])
    # TODO: we could make it lazy here, but we have some issues with dependencies using this variable, not VRE's one
    # which will instantiate them at __call__ time.
    # obj = partial(obj_type, video=video, name=name, dependencies=dependencies, **repr_cfg["parameters"])
    return obj

def build_representations_from_cfg(video: pims.Video,
                                   representations_dict: dict | DictConfig) -> dict[str, Representation]:
    if isinstance(representations_dict, DictConfig):
        representations_dict = OmegaConf.to_container(representations_dict, resolve=True)
    assert isinstance(representations_dict, dict), type(representations_dict)
    assert len(representations_dict) > 0 and isinstance(representations_dict, dict), representations_dict
    tsr: dict[str, Representation] = {}
    logger.debug("Doing topological sort...")
    dep_graph = {}
    for repr_name, repr_cfg_values in representations_dict.items():
        assert isinstance(repr_cfg_values, dict), f"{repr_name} not a dict cfg: {type(repr_cfg_values)}"
        dep_graph[repr_name] = repr_cfg_values["dependencies"]
    topo_sorted = {k: representations_dict[k] for k in topological_sort(dep_graph)}
    for name, r in topo_sorted.items():
        obj = build_representation_from_cfg(video, r, name, tsr)
        tsr[name] = obj
    return tsr
