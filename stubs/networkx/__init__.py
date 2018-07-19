from typing import Union, List, Tuple, Any, Optional, Dict

import numpy as np

class Graph:
    def __init__(
        self,
        data: Union[None, List[Tuple[Any, Any]], Graph, np.ndarray]
    ) -> None:
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
