from typing import Union, List, Tuple, Any, Optional, Dict, Hashable, Iterable, Iterator, Set, Callable

import numpy as np

class Graph:
    def __init__(
        self,
        data: Union[None, List[Tuple[Any, Any]], Graph, np.ndarray] = None
    ) -> None:
        ...

    def add_node(self, n: Hashable, attr_dict: Optional[dict] = None, **kwargs: Any) -> None:
        ...

    def add_edge(self, u: Hashable, v: Hashable, attr_dict: Optional[dict] = None, **kwargs: Any) -> None:
        ...

    def nodes(self, data: bool = False) -> List[Hashable]:
        ...

    def edges(self, nbunch: Optional[Iterable[Hashable]] = None, data: bool = False) -> List[Tuple[Hashable, Hashable]]:
        ...

    def subgraph(self, nbunch: Iterable[Hashable]) -> Graph:
        ...
        
class DiGraph(Graph):
    def subgraph(self, nbunch: Iterable[Hashable]) -> DiGraph:
        ...


def shortest_path(
    G: Graph,
    source: Any = None,
    target: Any = None,
    weight: Optional[str] = None
) -> Union[List, Dict[Any, List], Dict[Any, Dict[Any, List]]]:
    ...

def shortest_path_length(
    G: Graph,
    source: Any = None,
    target: Any = None,
    weight: Optional[str] = None
) -> Union[int, Dict[Any, int], Dict[Any, Dict[Any, int]]]:
    ...

def connected_components(G: Graph) -> Iterator[Set[Hashable]]:
    ...

def is_isomorphic(
        G1: Graph,
        G2: Graph,
        node_match: Optional[Callable[[dict, dict], bool]] = None,
        edge_match: Optional[Callable[[dict, dict], bool]] = None
) -> bool:
    ...

class NetworkXException(Exception):
    ...

class NetworkXError(NetworkXException):
    ...


class NetworkXPointlessConcept(NetworkXException):
    ...


class NetworkXAlgorithmError(NetworkXException):
    ...


class NetworkXUnfeasible(NetworkXAlgorithmError):
    ...


class NetworkXNoPath(NetworkXUnfeasible):
    ...


class NetworkXNoCycle(NetworkXUnfeasible):
    ...


class NetworkXUnbounded(NetworkXAlgorithmError):
    ...


class NetworkXNotImplemented(NetworkXException):
    ...
