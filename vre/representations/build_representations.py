"""external_representations.py handles functions related to loading external representations to the library"""
import imp # pylint: disable=deprecated-module
from pathlib import Path
from typing import Type
from omegaconf import DictConfig, OmegaConf
from .representation import Representation
from .io_representation_mixin import IORepresentationMixin
from .compute_representation_mixin import ComputeRepresentationMixin
from .learned_representation_mixin import LearnedRepresentationMixin
from ..logger import vre_logger as logger
from ..utils import topological_sort

def add_external_repositories(external_paths: list[str],
                              default_representations: dict[str, type[Representation]] | None = None) \
        -> dict[str, type[Representation]]:
    """adds external repositories from provided paths in the format: /path/to/script.py:fn_name"""
    representation_types = default_representations or {}
    for external_path in external_paths:
        external_types = _add_one_external_repository(external_path)
        assert (diff := set(representation_types.keys()).intersection(external_types)) == set(), diff
        representation_types = {**representation_types, **external_types}
    return representation_types

def _add_one_external_repository(external_path: str) -> dict[str, type[Representation]]:
    path, fn = external_path.split(":")
    external_reprs: dict[str, Representation] = getattr(imp.load_source("external", path), fn)()
    assert all(isinstance(v, Type) for v in external_reprs.values()), external_reprs
    return external_reprs

def _validate_repr_cfg(repr_cfg: dict, name: str):
    assert isinstance(repr_cfg, dict), f"Broken format (not a dict) for {name}. Type: {type(repr_cfg)}."
    valid_keys = {"type", "parameters", "dependencies", "compute_parameters", "learned_parameters", "io_parameters"}
    required_keys = {"type", "parameters", "dependencies"}
    assert set(repr_cfg).issubset(valid_keys), f"{name} wrong keys: {repr_cfg.keys()}"
    assert (diff := required_keys.difference(repr_cfg)) == set(), f"{name} missing keys: {diff}"
    assert isinstance(repr_cfg["parameters"], dict), type(repr_cfg["parameters"])
    assert name.find("/") == -1, "No '/' allowed in the representation name. Got '{name}'"

def build_representation_from_cfg(repr_cfg: dict, name: str, representation_types: dict[str, type[Representation]],
                                  built_so_far: list[Representation],
                                  compute_defaults: dict, learned_defaults: dict, io_defaults: dict) -> Representation:
    """Builds a representation given a dict config and a name."""
    assert isinstance(learned_defaults, dict) and isinstance(compute_defaults, dict) and isinstance(io_defaults, dict)
    _validate_repr_cfg(repr_cfg, name)

    logger.info(f"Building '{repr_cfg['type']}' (vre name: {name})")
    obj_type = representation_types[repr_cfg["type"]]
    built_so_far_dict = {r.name: r for r in built_so_far}
    dependencies = [built_so_far_dict[dep] for dep in repr_cfg["dependencies"]]
    obj: Representation = obj_type(name=name, dependencies=dependencies, **repr_cfg["parameters"])

    learned_params, compute_params, io_params = learned_defaults, compute_defaults, io_defaults
    if "learned_parameters" in repr_cfg:
        assert isinstance(obj, LearnedRepresentationMixin), obj
        learned_params = {**learned_params, **repr_cfg["learned_parameters"]}
        logger.debug(f"[{obj}] Setting node specific 'Learned' params: {learned_params}")
    if "compute_parameters" in repr_cfg:
        assert isinstance(obj, ComputeRepresentationMixin), obj
        compute_params = {**compute_params, **repr_cfg["compute_parameters"]}
        logger.debug(f"[{obj}] Setting node specific 'Compute' params: {compute_params}")
    if "io_parameters" in repr_cfg:
        assert isinstance(obj, IORepresentationMixin), obj
        io_params = {**io_params, **repr_cfg["io_parameters"]}
        logger.debug(f"[{obj}] Setting node specific 'IO' params: {io_params}")

    if isinstance(obj, ComputeRepresentationMixin):
        obj.set_compute_params(**compute_params)
    if isinstance(obj, LearnedRepresentationMixin):
        obj.set_learned_params(**learned_params)
    if isinstance(obj, IORepresentationMixin):
        obj.set_io_params(**io_params)
    return obj

def build_representations_from_cfg(cfg: Path | str | DictConfig | dict,
                                   representation_types: dict[str, type[Representation]],
                                   external_representations: list[Path] | None = None) -> list[Representation]:
    """builds a list of representations given a dict config (yaml file)"""
    assert isinstance(cfg, (Path, str, DictConfig, dict)), type(cfg)
    cfg = OmegaConf.load(cfg) if isinstance(cfg, (Path, str)) else cfg
    cfg: dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    assert len(repr_cfg := cfg["representations"]) > 0 and isinstance(repr_cfg, dict), repr_cfg

    logger.debug("Doing topological sort...")
    dep_graph = {repr_name: repr_cfg_values["dependencies"] for repr_name, repr_cfg_values in repr_cfg.items()}
    topo_sorted = {k: repr_cfg[k] for k in topological_sort(dep_graph)}

    compute_defaults = cfg.get("default_compute_parameters", {})
    learned_defaults = cfg.get("default_learned_parameters", {})
    io_defaults = cfg.get("default_io_parameters", {})

    built_so_far: list[Representation] = []
    for name, repr_cfg in topo_sorted.items():
        obj = build_representation_from_cfg(repr_cfg=repr_cfg, name=name, representation_types=representation_types,
                                            built_so_far=built_so_far, compute_defaults=compute_defaults,
                                            learned_defaults=learned_defaults, io_defaults=io_defaults)
        built_so_far.append(obj)

    for external_repr in (external_representations or []):
        built_so_far = _add_one_external_representation_list(built_so_far, external_repr,
                                                             compute_params=compute_defaults,
                                                             learned_params=learned_defaults, io_params=io_defaults)
    return built_so_far

def _add_external_representations_dict(built_so_far: list[Representation],
                                       external_reprs: dict[str, Representation],
                                       compute_params: dict, learned_params: dict,
                                       io_params: dict) -> list[Representation]:
    assert all(isinstance(v, Representation) for v in external_reprs.values()), external_reprs
    name_to_repr = {r.name: r for r in built_so_far}
    assert (diff := set(name_to_repr.keys()).intersection(external_reprs)) == set(), diff

    # Note (TODO?): ideally we'd use deepcopy here so we don't update external_reprs stuff, however there is an issue
    # if we do this, becvause there is no guarantee that external_reprs is topo-sorted (which we could do). For this
    # reason. we cannot do stuff like semantic_output (see semantic_mapper) that depends on 3 representations
    # (converted) which are also task mapped from the raw ones, because we'd update the dependencies of the middle ones
    # (converted) initially os it uses 'built_so_far' objects, but then it crashes on depth=2 (semantic_output).
    for obj in external_reprs.values():
        if isinstance(obj, ComputeRepresentationMixin):
            obj.set_compute_params(**compute_params)
        if isinstance(obj, LearnedRepresentationMixin):
            obj.set_learned_params(**learned_params)
        if isinstance(obj, IORepresentationMixin):
            obj.set_io_params(**io_params)

        # update clashes in dependencies
        for i, external_dep in enumerate(obj.dependencies):
            if external_dep.name in name_to_repr and id(external_dep) != id(curr := name_to_repr[external_dep.name]):
                logger.warning(f"[{obj.name}] Dependency {external_dep} is different than existing {curr}. "
                                "Replacing the dependency. This may yield in wrong results!")
                obj.dependencies[i] = curr
    return [*built_so_far, *external_reprs.values()]

def _add_one_external_representation_list(built_so_far: list[Representation], external_path: str, compute_params: dict,
                                          learned_params: dict, io_params: dict) -> list[Representation]:
    assert isinstance(learned_params, dict) and isinstance(compute_params, dict) and isinstance(io_params, dict)
    path, fn = external_path.split(":")
    external_reprs: dict[str, Representation] = getattr(imp.load_source("external", path), fn)()
    assert isinstance(external_reprs, dict) and len(external_reprs) > 0, (external_path, external_reprs)
    logger.info(f"Adding {list(external_reprs)} from {path}")
    return _add_external_representations_dict(built_so_far, external_reprs, compute_params, learned_params, io_params)
