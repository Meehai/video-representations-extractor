"""representations_list.py: module that implements a wrapper on top of a list of representations (auto sorting etc.)"""
from __future__ import annotations
from typing import Any, Iterator
from overrides import overrides
from vre.utils import vre_topo_sort
from .representation import Representation
from .io_representation_mixin import IORepresentationMixin

class RepresentationsList(list):
    """module that topo-sorts and implements helper functions on top of a representation list"""
    def __init__(self, representations: list[Representation] | RepresentationsList, topo_sort: bool = True):
        representations = vre_topo_sort(representations) if topo_sort else representations
        super().__init__(representations)
        assert len(self) > 0, "At least one representation must be provided"
        assert all(isinstance(x, Representation) for x in self), [(x.name, type(x)) for x in self]
        self.names = [r.name for r in self]

    @overrides
    def append(self, object: Any): # pylint: disable=redefined-builtin
        raise ValueError("Cannot append, use the constructor")

    def get_output_representations(self, subset: list[str] | None = None) -> RepresentationsList:
        """
        Given all the representations, keep those that actually export something. RunMetadata uses only these.
        We do a bunch of checks:
        - at least one representation is provided
        - they are exportable: IORpresentations are set, like npz, png etc.
        - no duplicates provided
        Returns a subset of representations from the ones given at constructor that will be exported.
        """
        subset = subset or self.names
        assert all(name in self.names for name in subset), (
            f"{subset=}\n{self.names=}\nmissing={[n for n in subset if n not in self.names]}")

        io_reprs: list[IORepresentationMixin] = [_r for _r in self if isinstance(_r, IORepresentationMixin)]
        res = []
        for r in io_reprs:
            if (r.export_binary or r.export_image) and r.name in subset:
                res.append(r)
        return RepresentationsList(res, topo_sort=False)

    def to_graphviz(self, **kwargs) -> "Digraph":
        """Returns a graphviz object from this graph. Used for plotting the graph. Best for smaller graphs."""
        from graphviz import Digraph # pylint: disable=import-outside-toplevel
        g = Digraph()
        for k, v in kwargs.items():
            g.attr(**{k: v})
        g.attr(rankdir="LR")
        for node in self:
            g.node(name=f"{node.name}", shape="oval")
        edges: list[tuple[str, str]] = [(r.name, dep.name) for r in self for dep in r.dependencies]
        for l, r in edges:
            g.edge(r, l) # reverse?
        return g

    @overrides
    def __iter__(self) -> Iterator[Representation]: # for type-hinting
        return super().__iter__()

    def __str__(self) -> str:
        return f"Representations ({len(self)}): [{', '.join(self.names)}]"

    def __repr__(self):
        return str(self)
