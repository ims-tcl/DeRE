from dere.corpus import Corpus
from dere.models import Model
from dere import Result


class PGModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> Corpus:
        ...

    def eval(self, corpus: Corpus, predicted: Corpus) -> Result:
        ...
