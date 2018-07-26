from dere.corpus import Corpus
from dere.corpus_io import CorpusIO


class XML123CorpusIO(CorpusIO):
    def load(self, path: str) -> Corpus:
        ...

    def dump(self, corpus: Corpus, path: str) -> None:
        ...
