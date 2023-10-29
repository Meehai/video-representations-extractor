"""Build representation module. Updates faster than README :)"""
from typing import Type
from omegaconf import DictConfig, OmegaConf
import pims
from .representation import Representation
from ..logger import logger


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
        elif method == "depth-dispresnet":
            from .depth.dispresnet import DepthDispResNet

            objType = DepthDispResNet

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


def build_representation_from_cfg(video: pims.Video, repr_cfg: DictConfig, name: str,
                                  built_so_far: dict[str, Representation]) -> Representation:
    logger.debug(f"Representation='{name}'. Instantiating...")
    assert isinstance(repr_cfg, DictConfig), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    assert "type" in repr_cfg and "method" in repr_cfg, f"Broken format: {repr_cfg.keys()}"
    repr_type, repr_method = repr_cfg["type"], repr_cfg["method"]
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    assert isinstance(repr_cfg["parameters"], DictConfig), type(repr_cfg["parameters"])

    obj_type = build_representation_type(repr_type, repr_method)
    # this is here because omegaconf transforms [1, 2, 3, 4] in a ListConfig, not a simple list
    # obj_args = OmegaConf.to_container(repr_cfg["parameters"])
    obj = obj_type(video=video, name=name, dependencies=dependencies, **repr_cfg["parameters"])
    return obj
