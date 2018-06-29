from dere import Corpus
from dere.readers import CorpusReader

class BRATCorpusReader(CorpusReader):
    def load(self) -> Corpus:
        ...
