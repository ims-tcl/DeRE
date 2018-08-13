from typing import Union, Iterable, List, Optional
_Label = Union[int, str]

# Technically this is too permissive, but good for the time being I think...
def flat_classification_report(
    y_true: Iterable[Iterable[_Label]],
    y_pred : Iterable[Iterable[_Label]],
    labels: Optional[Iterable[_Label]] = None,
    target_names: Optional[List[str]] = None,
    sample_weight: Optional[Iterable[float]] = None,
    digits: Optional[int] = None
) -> str:
    ...

def flat_f1_score(
    y_true: Iterable[Iterable[_Label]],
    y_pred : Iterable[Iterable[_Label]],
    labels: Optional[Iterable[_Label]] = None,
    pos_label: Union[str, int] = 1,
    average: Optional[str] = 'binary',
    sample_weight: Optional[Iterable[float]] = None
) -> float:
    ...
