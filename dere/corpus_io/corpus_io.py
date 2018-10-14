import logging
from pathlib import Path
from typing import Union

from dere.corpus import Corpus
from dere.taskspec import TaskSpecification


class CorpusIO:
    def __init__(self, spec: TaskSpecification) -> None:
        self._spec = spec
        self._logger = logging.getLogger(__name__)

    def load(self, path: str, load_gold: bool = True) -> Corpus:
        ...

    def dump(self, corpus: Corpus, path: str, just_predictions: bool = True) -> None:
        ...
