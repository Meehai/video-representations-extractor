"""topological sort module"""
from typing import TypeVar

T = TypeVar("T")

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

def vre_topo_sort(representations: list["Representation"]) -> list["Representation"]:
    """does a topological sort of a list of representations. They must have .name and .deps"""
    dep_graph = {r.name: [_r.name for _r in r.dependencies] for r in representations}
    tsr = topological_sort(dep_graph)

    # do a DFS to ensure that all the representations of the same are the same object!
    # guaranteed to have no cycles due to the topo sort above
    repr_to_id = {r.name: id(r) for r in representations}
    to_process = [*representations]
    while len(to_process) > 0:
        r = to_process.pop()
        if r.name in repr_to_id:
            if (id1 := id(r)) != (id2 := repr_to_id[r.name]):
                raise ValueError(f"Representation '{r}' has a different id: {id1} than previously provided: {id2}")
            to_process.extend(r.dependencies)
        else:
            repr_to_id[r.name] = id(r)

    name_to_repr = {r.name: r for r in representations}
    return [name_to_repr[r_name] for r_name in tsr]
