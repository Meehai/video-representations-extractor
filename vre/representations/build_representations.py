"""builder for all the representations in VRE. Can be used outside of VRE too, see examples/notebooks."""
from typing import Type
from omegaconf import DictConfig, OmegaConf

from ..logger import vre_logger as logger
from ..utils import topological_sort
from .representation import Representation
from .learned_representation_mixin import LearnedRepresentationMixin
from .compute_representation_mixin import ComputeRepresentationMixin

# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-branches, too-many-statements, cyclic-import
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

def build_representation_from_cfg(repr_cfg: dict, name: str, built_so_far: dict[str, Representation],
                                  compute_representations_defaults: dict | None = None,
                                  learned_representations_defaults: dict | None = None) -> Representation:
    """Builds a representation given a dict config and a name."""
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    assert set(repr_cfg).issubset({"type", "parameters", "dependencies", "compute_parameters", "learned_parameters"}), \
        f"{name} wrong keys: {repr_cfg.keys()}"
    assert {"type", "parameters", "dependencies"}.issubset(repr_cfg), f"{name} missing keys: {repr_cfg.keys()}"
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])
    assert name.find("/") == -1, "No '/' allowed in the representation name. Got '{name}'"

    logger.info(f"Building '{repr_cfg['type']}' (vre name: {name})")
    obj_type = build_representation_type(repr_cfg["type"])
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    obj: Representation = obj_type(name=name, dependencies=dependencies, **repr_cfg["parameters"])

    if isinstance(obj, LearnedRepresentationMixin):
        defaults = {} if learned_representations_defaults is None else learned_representations_defaults
        defaults = OmegaConf.to_container(defaults, resolve=True) if isinstance(defaults, DictConfig) else defaults
        if "learned_parameters" in repr_cfg:
            repr_learned_parameters = {**defaults, **repr_cfg.get("learned_parameters", {})}
            logger.debug(f"[{obj}] Setting node specific params: {repr_learned_parameters}")
        else:
            repr_learned_parameters = defaults
            logger.debug(f"[{obj}] Setting default params: {repr_learned_parameters}")
        obj.set_learned_params(**repr_learned_parameters)
    else:
        assert "learned_parameters" not in repr_cfg, f"Learned parameters not allowed for {name}"

    if isinstance(obj, ComputeRepresentationMixin):
        defaults = {} if compute_representations_defaults is None else compute_representations_defaults
        defaults = OmegaConf.to_container(defaults, resolve=True) if isinstance(defaults, DictConfig) else defaults
        if "compute_parameters" in repr_cfg:
            repr_compute_params = {**defaults, **repr_cfg.get("compute_parameters", {})}
            logger.debug(f"[{obj}] Setting node specific params: {repr_compute_params}")
        else:
            repr_compute_params = defaults
            logger.debug(f"[{obj}] Setting default params: {repr_compute_params}")
        obj.set_compute_params(**repr_compute_params)
    else:
        assert "compute_parameters" not in repr_cfg, f"Compute parameters not allowed for {name}"
    return obj

def build_representations_from_cfg(representations_dict: dict | DictConfig,
                                   compute_representations_default: dict | None = None,
                                   learned_representations_defaults: dict | None = None ) -> dict[str, Representation]:
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
        obj = build_representation_from_cfg(repr_cfg, name, tsr, compute_representations_default,
                                            learned_representations_defaults)
        tsr[name] = obj
    return tsr
