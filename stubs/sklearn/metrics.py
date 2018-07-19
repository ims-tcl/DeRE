from typing import List, Optional, TypeVar, Union, Tuple

from sklearn import _ArrayLike

def accuracy_score(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
    normalize: bool = True,
    sample_weight: Optional[_ArrayLike] = None
) -> float:
    ...


def precision_recall_fscore_support(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
    beta: float = 1.0,
    labels: Optional[List] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = None,
    sample_weight: Optional[_ArrayLike] = None
) -> Tuple[
    Union[float, _ArrayLike],
    Union[float, _ArrayLike],
    Union[float, _ArrayLike],
    Union[int, _ArrayLike]
]:
    ...


def confusion_matrix(
    y_true: _ArrayLike,
    y_pred: _ArrayLike,
    labels: Optional[List] = None,
    sample_weight: Optional[_ArrayLike] = None
) -> _ArrayLike:
    ...
