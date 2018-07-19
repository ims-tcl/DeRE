from typing import Sequence, Optional, Tuple, Union
from numpy import dtype, ndarray

class spmatrix:
    nnz: int
    shape: Tuple[int, ...]

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

