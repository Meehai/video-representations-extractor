"""builder for all the representations in VRE. Can be used outside of VRE too, see examples/notebooks."""
from typing import Type
from omegaconf import DictConfig, OmegaConf
from ..logger import vre_logger as logger
from ..utils import topological_sort
from ..representation import Representation

# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-branches, too-many-statements
def build_representation_type(repr_type: str) -> Type[Representation]:
    """Gets the representation type from a type and a name (the two identifiers of a representation)"""
    obj_type: Type[Representation] | None = None
    assert len(repr_type.split("/")) == 2, f"No extra '/' allowed in the type. Got: '{repr_type}'. Correct: depth/dpt"

    if repr_type == "default/rgb":
        from .rgb import RGB
        obj_type = RGB
    elif repr_type == "default/hsv":
        from .hsv import HSV
        obj_type = HSV
    elif repr_type == "soft-segmentation/python-halftone":
        from .soft_segmentation.halftone import Halftone
        obj_type = Halftone
    elif repr_type == "soft-segmentation/generalized_boundaries":
        from .soft_segmentation.generalized_boundaries import GeneralizedBoundaries
        obj_type = GeneralizedBoundaries
    elif repr_type == "soft-segmentation/fastsam":
        from .soft_segmentation.fastsam import FastSam
        obj_type = FastSam
    elif repr_type == "edges/dexined":
        from .edges.dexined import DexiNed
        obj_type = DexiNed
    elif repr_type == "edges/canny":
        from .edges.canny import Canny
        obj_type = Canny
    elif repr_type == "depth/dpt":
        from .depth.dpt import DepthDpt
        obj_type = DepthDpt
    elif repr_type == "depth/marigold":
        from .depth.marigold import Marigold
        obj_type = Marigold
    elif repr_type == "optical-flow/rife":
        from .optical_flow.rife import FlowRife
        obj_type = FlowRife
    elif repr_type == "optical-flow/raft":
        from .optical_flow.raft import FlowRaft
        obj_type = FlowRaft
    elif repr_type == "semantic-segmentation/safeuav":
        from .semantic_segmentation.safeuav import SafeUAV
        obj_type = SafeUAV
    elif repr_type == "semantic-segmentation/mask2former":
        from .semantic_segmentation.mask2former import Mask2Former
        obj_type = Mask2Former
    elif repr_type == "normals/depth-svd":
        from .normals.depth_svd import DepthNormalsSVD
        obj_type = DepthNormalsSVD
    else:
        raise ValueError(f"Unknown type: '{repr_type}'")
    return obj_type

def build_representation_from_cfg(repr_cfg: dict, name: str, built_so_far: dict[str, Representation]) -> Representation:
    """
    Builds a representation given a dict config and a name.
    Convention:
    - name: VRE Name aka runtime name or how the directories are going to be called after vre call
    - repr_cfg["type"] The type of the representation (i.e optical-flow/rife, semanatic-segmentation/mask2former,
        depth/dpt etc.)
    - parameters The parameters sent to each representation
    - dependencies The dependencies names based for each representation (can be an empty list)
    """
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    assert {"type", "parameters", "dependencies"}.issubset(repr_cfg), f"{name} wrong keys: {repr_cfg.keys()}"
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])
    assert name.find("/") == -1, "No '/' allowed in the representation name. Got '{name}'"
    logger.info(f"Building '{repr_cfg['type']}' (vre name: {name})")
    obj_type = build_representation_type(repr_cfg["type"])
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    obj: Representation = obj_type(name=name, dependencies=dependencies, **repr_cfg["parameters"])

    assert "vre_parameters" not in repr_cfg, "Old config file, remove 'vre_parameters'"
    if "device" in repr_cfg:
        logger.info(f"Explicit device provided: {repr_cfg['device']}. This device will be used at vre.run()")
        obj.device = repr_cfg["device"]
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
