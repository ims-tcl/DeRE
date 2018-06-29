from dere import Corpus, Result
from dere.models import Model

class PGModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    def eval(self, corpus: Corpus, predicted: list) -> Result:
        ...
