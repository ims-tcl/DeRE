from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Result:
    precision: float
    recall: float

    def __sub__(self, other: Result) -> Result:
        # this should work now in Python 3.7
        return Result(self.precision - other.precision, self.recall - other.recall)
