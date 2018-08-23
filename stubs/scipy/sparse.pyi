from typing import Sequence, Optional, Tuple, Union, Iterator
from numpy import dtype, ndarray

class spmatrix:
    nnz: int
    shape: Tuple[int, ...]
    def __iter__(self) -> Iterator:
        ...

class csr_matrix(spmatrix):
    def __init__(
        self,
        arg1: Union[ndarray, spmatrix]
    ) -> None:
        ...

def hstack(
    blocks: Sequence[spmatrix],
    format: Optional[str] = None,
    dtype: Optional[dtype] = None
) -> spmatrix:
    ...

def vstack(
    blocks: Sequence[spmatrix],
    format: Optional[str] = None,
    dtype: Optional[dtype] = None
) -> spmatrix:
    ...

