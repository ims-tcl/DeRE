import logging
from pathlib import Path

from dere.corpus import Corpus
from dere.schema import TaskSchema


class CorpusReader:
    def __init__(self, corpus_path: str, schema: TaskSchema) -> None:
        self._corpus_path = Path(corpus_path)
        self._schema = schema
        self._logger = logging.getLogger(__name__)

    def load(self) -> Corpus:
        ...
