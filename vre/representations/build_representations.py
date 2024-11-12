"""builder for all the representations in VRE. Can be used outside of VRE too, see examples/notebooks."""
from typing import Type
import imp # pylint: disable=deprecated-module
from omegaconf import DictConfig, OmegaConf

from ..logger import vre_logger as logger
from ..utils import topological_sort
from .representation import Representation
from .learned_representation_mixin import LearnedRepresentationMixin
from .compute_representation_mixin import ComputeRepresentationMixin
from .io_representation_mixin import IORepresentationMixin

# pylint: disable=import-outside-toplevel, redefined-builtin, too-many-branches, too-many-statements, cyclic-import
def build_representation_type(repr_type: str) -> Type[Representation]:
    """Gets the representation type from a type and a name (the two identifiers of a representation)"""
    obj_type: Type[Representation] | None = None
    assert len(repr_type.split("/")) == 2, f"No extra '/' allowed in the type. Got: '{repr_type}'. Correct: depth/dpt"

    if repr_type == "default/rgb":
        from .color.rgb import RGB
        obj_type = RGB
    elif repr_type == "default/hsv":
        from .color.hsv import HSV
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

def _validate_repr_cfg(repr_cfg: dict, name: str):
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    valid_keys = {"type", "parameters", "dependencies", "compute_parameters", "learned_parameters", "io_parameters"}
    required_keys = {"type", "parameters", "dependencies"}
    assert set(repr_cfg).issubset(valid_keys), f"{name} wrong keys: {repr_cfg.keys()}"
    assert (diff := required_keys.difference(repr_cfg)) == set(), f"{name} missing keys: {diff}"
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])
    assert name.find("/") == -1, "No '/' allowed in the representation name. Got '{name}'"

def build_representation_from_cfg(repr_cfg: dict, name: str, built_so_far: dict[str, Representation],
                                  compute_representations_defaults: dict | None = None,
                                  learned_representations_defaults: dict | None = None,
                                  io_representations_defaults: dict | None = None) -> Representation:
    """Builds a representation given a dict config and a name."""
    _validate_repr_cfg(repr_cfg, name)
    logger.info(f"Building '{repr_cfg['type']}' (vre name: {name})")
    obj_type = build_representation_type(repr_cfg["type"])
    dependencies = [built_so_far[dep] for dep in repr_cfg["dependencies"]]
    obj: Representation = obj_type(name=name, dependencies=dependencies, **repr_cfg["parameters"])

    if isinstance(obj, LearnedRepresentationMixin):
        defaults = {} if learned_representations_defaults is None else learned_representations_defaults
        defaults = OmegaConf.to_container(defaults, resolve=True) if isinstance(defaults, DictConfig) else defaults
        if "learned_parameters" in repr_cfg:
            repr_learned_parameters = {**defaults, **repr_cfg.get("learned_parameters", {})}
            logger.debug(f"[{obj}] Setting node specific 'Learned' params: {repr_learned_parameters}")
        else:
            repr_learned_parameters = defaults
            logger.debug(f"[{obj}] Setting default 'Learned' params: {repr_learned_parameters}")
        obj.set_learned_params(**repr_learned_parameters)
    else:
        assert "learned_parameters" not in repr_cfg, f"Learned parameters not allowed for {name}"

    if isinstance(obj, ComputeRepresentationMixin):
        defaults = {} if compute_representations_defaults is None else compute_representations_defaults
        defaults = OmegaConf.to_container(defaults, resolve=True) if isinstance(defaults, DictConfig) else defaults
        if "compute_parameters" in repr_cfg:
            repr_compute_params = {**defaults, **repr_cfg.get("compute_parameters", {})}
            logger.debug(f"[{obj}] Setting node specific 'Compute' params: {repr_compute_params}")
        else:
            repr_compute_params = defaults
            logger.debug(f"[{obj}] Setting default 'Compute' params: {repr_compute_params}")
        obj.set_compute_params(**repr_compute_params)
    else:
        assert "compute_parameters" not in repr_cfg, f"Compute parameters not allowed for {name}"

    if isinstance(obj, IORepresentationMixin):
        defaults = {} if io_representations_defaults is None else io_representations_defaults
        defaults = OmegaConf.to_container(defaults, resolve=True) if isinstance(defaults, DictConfig) else defaults
        if "io_parameters" in repr_cfg:
            repr_io_params = {**defaults, **repr_cfg.get("io_parameters", {})}
            logger.debug(f"[{obj}] Setting node specific 'I/O' params: {repr_io_params}")
        else:
            repr_io_params = defaults
            logger.debug(f"[{obj}] Setting default 'I/O' params: {repr_io_params}")
        obj.set_io_params(**repr_io_params)
    else:
        assert "io_parameters" not in repr_cfg, f"I/O parameters not allowed for {name}"
    return obj

def build_representations_from_cfg(cfg: DictConfig | dict) -> dict[str, Representation]:
    """builds a dict of representations given a dict config (yaml file)"""
    cfg: dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    assert len(repr_cfg := cfg["representations"]) > 0 and isinstance(repr_cfg, dict), repr_cfg

    tsr: dict[str, Representation] = {}
    logger.debug("Doing topological sort...")
    dep_graph = {}
    for repr_name, repr_cfg_values in repr_cfg.items():
        _validate_repr_cfg(repr_cfg_values, repr_name)
        assert isinstance(repr_cfg_values, dict), f"{repr_name} not a dict cfg: {type(repr_cfg_values)}"
        dep_graph[repr_name] = repr_cfg_values["dependencies"]
    topo_sorted = {k: repr_cfg[k] for k in topological_sort(dep_graph)}

    for name, repr_cfg in topo_sorted.items():
        obj = build_representation_from_cfg(repr_cfg, name, tsr, cfg.get("default_compute_parameters"),
                                            cfg.get("default_learned_parameters"), cfg.get("default_io_parameters"))
        tsr[name] = obj
    return tsr

def add_external_representations(representations: dict[str, Representation], external_path: str,
                                 cfg: DictConfig) -> dict[str, Representation]:
    """adds external representations from an provided path in the format: /path/to/script.py:fn_name"""
    path, fn = external_path.split(":")
    external_representations: dict[str, Representation] = getattr(imp.load_source("external", path), fn)()
    assert all(isinstance(v, IORepresentationMixin) for v in external_representations.values())
    assert all(isinstance(v, ComputeRepresentationMixin) for v in external_representations.values())
    assert all(isinstance(v, Representation) for v in external_representations.values())
    assert (clash := set(external_representations.keys()).intersection(representations)) == set(), clash
    logger.info(f"Adding {list(external_representations)} from {path}")
    for repr in external_representations.values():
        repr.set_compute_params(**cfg.get("default_compute_parameters", {}))
        repr.set_io_params(**cfg.get("default_io_parameters", {}))
        if isinstance(repr, LearnedRepresentationMixin):
            repr.set_learned_parameters(**cfg.get("default_learned_parameters", {}))
    representations = {**representations, **external_representations}
    tsr = topological_sort({r.name: [_r.name for _r in r.dependencies] for r in representations.values()})
    representations = {k: representations[k] for k in tsr}
    return representations
