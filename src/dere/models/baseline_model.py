from dere.corpus import Corpus
from dere.models import Model
from dere import Result


class BaselineModel(Model):
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...

