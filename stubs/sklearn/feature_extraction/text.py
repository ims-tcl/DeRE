from typing import Iterable, Union
from io import TextIOBase

import scipy.sparse
from sklearn import _ArrayLike

class CountVectorizer:
    def fit(
        self,
        raw_documents: Iterable[Union[str, TextIOBase]]
    ) -> _ArrayLike:
        ...

    def transform(
        self,
        raw_documents: Iterable[Union[str, TextIOBase]]
    ) -> scipy.sparse.spmatrix:
        ...
