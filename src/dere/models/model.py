from dere.corpus import Corpus
from dere.schema import TaskSchema
from dere import Result


class Model:
    def __init__(self, schema: TaskSchema) -> None:
        ...

    # only minimal logic here, things that all models (might) need
    def train(self, corpus: Corpus) -> None:
        ...

    def predict(self, corpus: Corpus) -> list:
        ...

    def eval(self, corpus: Corpus, predicted: list) -> Result:
        ...  # here there might actually be a sensible default implementation
