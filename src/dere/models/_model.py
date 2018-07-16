from dere.corpus import Corpus
from dere.taskspec import TaskSpecification
from dere import Result


class Model:
    def __init__(self, spec: TaskSpecification) -> None:
        self.spec = spec

    # only minimal logic here, things that all models (might) need
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> None:
        ...

    def eval(self, corpus: Corpus, predicted: Corpus) -> Result:
        ...  # here there might actually be a sensible default implementation
