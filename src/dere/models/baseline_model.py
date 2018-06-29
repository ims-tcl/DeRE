from dere import Corpus, Result
from dere.models import Model


class BaselineModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...

