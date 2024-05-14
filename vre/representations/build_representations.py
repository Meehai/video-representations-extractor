"""builder for all the representations in VRE. Can be used outside of VRE too, see examples/notebooks."""
from typing import Type
from omegaconf import DictConfig, OmegaConf
from ..logger import logger
from ..utils import topological_sort
from ..representation import Representation

# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-branches, too-many-statements
def build_representation_type(type: str, name: str) -> Type[Representation]:
    """Gets the representation type from a type and a name (the two identifiers of a representation)"""
    obj_type: Type[Representation] | None = None

    if type == "default":
        if name == "rgb":
            from .rgb import RGB
            obj_type = RGB
        elif name == "hsv":
            from .hsv import HSV
            obj_type = HSV

    elif type == "soft-segmentation":
        if name == "python-halftone":
            from .soft_segmentation.halftone import Halftone
            obj_type = Halftone
        elif name == "kmeans":
            from .soft_segmentation.kmeans import KMeans
            obj_type = KMeans
        elif name == "generalized_boundaries":
            from .soft_segmentation.generalized_boundaries import GeneralizedBoundaries
            obj_type = GeneralizedBoundaries
        elif name == "fastsam":
            from .soft_segmentation.fastsam import FastSam
            obj_type = FastSam

    elif type == "edges":
        if name == "dexined":
            from .edges.dexined import DexiNed
            obj_type = DexiNed
        elif name == "canny":
            from .edges.canny import Canny
            obj_type = Canny

    elif type == "depth":
        if name == "dpt":
            from .depth.dpt import DepthDpt
            obj_type = DepthDpt
        elif name == "odo-flow":
            from .depth.odo_flow import DepthOdoFlow
            obj_type = DepthOdoFlow

    elif type == "optical-flow":
        if name == "rife":
            from .optical_flow.rife import FlowRife
            obj_type = FlowRife
        elif name == "raft":
            from .optical_flow.raft import FlowRaft
            obj_type = FlowRaft

    elif type == "semantic-segmentation":
        if name == "safeuav":
            from .semantic_segmentation.safeuav import SafeUAV
            obj_type = SafeUAV
        elif name == "mask2former":
            from .semantic_segmentation.mask2former import Mask2Former
            obj_type = Mask2Former

    elif type == "normals":
        if name == "depth-svd":
            from .normals.depth_svd import DepthNormalsSVD
            obj_type = DepthNormalsSVD

    assert obj_type is not None, f"Unknown type: {type}, name: {name}"
    return obj_type

def build_representation_from_cfg(repr_cfg: dict, name: str, built_so_far: dict[str, Representation]) -> Representation:
    """
    Builds a representation given a dict config and a name.
    Convention:
    - name: VRE Name aka runtime name or how the directories are going to be called after vre call
    - repr_cfg["type"] The type of the representation (i.e optical flow, semanatic, depth etc.)
    - repr_cfg["name"] The name of the type of representation (i.e rife, dpt, m2f etc.)
    """
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    assert {"type", "name", "parameters", "dependencies"}.issubset(repr_cfg), f"{name} wrong keys: {repr_cfg.keys()}"
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])

    logger.info(f"Building '{repr_cfg['type']}'/'{repr_cfg['name']}' (vre name: {name})")
    obj_type = build_representation_type(repr_cfg["type"], repr_cfg["name"])
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    obj: Representation = obj_type(name=name, dependencies=dependencies, **repr_cfg["parameters"])

    if "vre_parameters" in repr_cfg:
        logger.info(f"Explicit VRE parameters provided to '{name}'")
        assert isinstance(repr_cfg["vre_parameters"], dict), type(repr_cfg["vre_parameters"])
        obj.vre_parameters = repr_cfg["vre_parameters"]
    if "batch_size" in repr_cfg:
        logger.info(f"Explicit batch size {repr_cfg['batch_size']} provided to {name}.")
        assert isinstance(repr_cfg["batch_size"], int), type(repr_cfg["batch_size"])
        obj.batch_size = repr_cfg["batch_size"]
    return obj

def build_representations_from_cfg(representations_dict: dict | DictConfig) -> dict[str, Representation]:
    """builds a dict of representations given a dict config (yaml file)"""
    if isinstance(representations_dict, DictConfig):
        representations_dict: dict = OmegaConf.to_container(representations_dict, resolve=True)
    assert isinstance(representations_dict, dict), type(representations_dict)
    assert len(representations_dict) > 0 and isinstance(representations_dict, dict), representations_dict
    tsr: dict[str, Representation] = {}
    logger.debug("Doing topological sort...")
    dep_graph = {}
    for repr_name, repr_cfg_values in representations_dict.items():
        assert isinstance(repr_cfg_values, dict), f"{repr_name} not a dict cfg: {type(repr_cfg_values)}"
        dep_graph[repr_name] = repr_cfg_values["dependencies"]
    topo_sorted = {k: representations_dict[k] for k in topological_sort(dep_graph)}
    for name, repr_cfg in topo_sorted.items():
        obj = build_representation_from_cfg(repr_cfg, name, tsr)
        tsr[name] = obj
    return tsr
