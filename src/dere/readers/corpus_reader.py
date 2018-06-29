from dere import Corpus
class CorpusReader:
    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = corpus_path

    def load(self) -> Corpus:
        ...
