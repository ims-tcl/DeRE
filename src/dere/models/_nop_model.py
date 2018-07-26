from dere.models import Model
from dere.corpus import Corpus
from dere.taskspec import TaskSpecification


class NOPModel(Model):
    def __init__(self, spec: TaskSpecification) -> None:
        self.spec = spec

    def train(self, corpus: Corpus) -> None:
        pass

    def predict(self, corpus: Corpus) -> None:
        pass
