from dere import Corpus
from dere.readers import  CorpusReader
class XML123CorpusReader(CorpusReader):
    def load(self) -> Corpus:
        ...
