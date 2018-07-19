from typing import Optional, Union, Dict, Any

import numpy as np

from sklearn import _ArrayLike

class LinearSVC:
    def __init__(
        self,
        penalty: str = 'l2',
        loss: str = 'squared_hinge',
        dual: bool = True,
        tol: float = 1e-4,
        C: float = 1.0,
        multi_class: str = 'ovr',
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Union[None, Dict[Any, float], str] = None,
        verbose: int = 0,
        random_state: Union[None, np.random.RandomState, int] = None,
        max_iter: int = 1000
    ) -> None:
        ...

    def fit(
        self,
        X: _ArrayLike,
        y: _ArrayLike,
        sample_weight: Optional[_ArrayLike] = None
    ) -> LinearSVC:
        ...

    def predict(
        self,
        X: _ArrayLike
    ) -> _ArrayLike:
        ...
