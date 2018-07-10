from dere.corpus import Corpus
from dere.models import Model
from dere.schema import TaskSchema
from dere import Result

# from .span_classifier import SpanClassifier
# from .slot_classifier import SlotClassifier


class BaselineModel(Model):
    def __init__(self, schema: TaskSchema) -> None:
        self.schema = schema
        # self._span_classifier = SpanClassifier(schema)
        # self._slot_classifier = SlotClassifier(schema)

    def train(self, corpus: Corpus) -> None:
        # self._span_classifier.train(corpus)
        # self._slot_classifier.train(corpus)
        ...

    def predict(self, corpus: Corpus) -> Corpus:
        return corpus

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...
