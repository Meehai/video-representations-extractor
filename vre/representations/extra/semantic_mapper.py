"""SemanticMapper module. Transforms one semantic segmentation into another."""
from __future__ import annotations
from typing import Callable
import numpy as np
from overrides import overrides

from ..representation import Representation, ReprOut
from ...utils import colorize_semantic_segmentation
from ...logger import vre_logger as logger

class SemanticMapper(Representation):
    """SemanticMapper implementation. Support for 1 or more underlying semantic segmentations + a merge function."""
    def __init__(self, original_classes: list[list[str]], mapping: list[dict[str, list[str]]],
                 color_map: list[tuple[int, int, int]],
                 merge_fn: Callable[[list[np.ndarray]], np.ndarray] | None = None, **kwargs):
        super().__init__(**kwargs)
        assert len(self.dependencies) >= 1, "No dependencies provided. Need at least one semantic segmentation to map."
        assert isinstance(mapping, list), type(mapping)
        assert len(mapping) == (B := len(self.dependencies)), (len(mapping), B)
        assert (A := len(original_classes)) == len(self.dependencies), (A, B)
        merge_fn = SemanticMapper._default_merge_fn if merge_fn is None else merge_fn
        self.original_classes = original_classes
        self.mapping = mapping
        self.color_map = color_map
        self.merge_fn = merge_fn
        unique_mapped = set()
        for m in mapping:
            unique_mapped.update(m.keys())
        assert len(unique_mapped) == len(color_map), (len(unique_mapped), len(color_map))

    def _make_one(self, semantic_dep_data: np.ndarray, mapping: dict[str, list[str]],
                  original_classes: list[str]) -> np.ndarray:
        mapping_ix = {list(mapping.keys()).index(k): [original_classes.index(_v)
                                                      for _v in v] for k, v in mapping.items()}
        flat_mapping = {}
        for k, v in mapping_ix.items():
            for _v in v:
                flat_mapping[_v] = k
        mapped_data = np.vectorize(flat_mapping.get)(semantic_dep_data).astype(np.uint8)
        return mapped_data

    @overrides
    def make(self, frames: np.ndarray, dep_data: dict[str, ReprOut] | None = None) -> ReprOut:
        res = []
        for dep, mapping, original_classes in zip(self.dependencies, self.mapping, self.original_classes):
            res.append(self._make_one(dep_data[dep.name].output, mapping, original_classes))
        merged = self.merge_fn(res)
        return ReprOut(output=merged)

    @overrides
    def make_images(self, frames: np.ndarray, repr_data: ReprOut) -> np.ndarray:
        return colorize_semantic_segmentation(repr_data.output, self.color_map)

    @overrides
    def resize(self, repr_data: ReprOut, new_size: tuple[int, int]) -> ReprOut:
        raise NotImplementedError

    @overrides
    def size(self, repr_data: ReprOut) -> tuple[int, int]:
        raise NotImplementedError

    @staticmethod
    def _default_merge_fn(data: list[np.ndarray]) -> np.ndarray:
        """the default merge fn. Keeps first entry only"""
        if len(data) > 1:
            logger.warning(f"Got {len(data)} mappings. Returning the first mapping only. Updated merge_fn in ctor!")
        return data[0]
