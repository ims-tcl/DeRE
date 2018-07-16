import logging
from pathlib import Path

from dere.corpus import Corpus
from dere.taskspec import TaskSpecification


class CorpusReader:
    def __init__(self, corpus_path: str, spec: TaskSpecification) -> None:
        self._corpus_path = Path(corpus_path)
        self._spec = spec
        self._logger = logging.getLogger(__name__)

    def load(self) -> Corpus:
        ...
