from typing import Optional

from dere.corpus import Corpus
from dere.models import Model
from dere.taskspec import TaskSpecification
from dere import Result

import numpy as np

from .span_classifier import SpanClassifier
from .slot_classifier import SlotClassifier


class BaselineModel(Model):
    def __init__(self, spec: TaskSpecification) -> None:
        self.spec = spec
        self._span_classifier = SpanClassifier(spec)
        self._slot_classifier = SlotClassifier(spec)

    def train(self, corpus: Corpus, dev_corpus: Optional[Corpus]) -> None:
        self._span_classifier.train(corpus, dev_corpus=dev_corpus)
        self._slot_classifier.train(corpus, dev_corpus=dev_corpus)

    def predict(self, corpus: Corpus) -> None:
        self._span_classifier.predict(corpus)
        self._slot_classifier.predict(corpus)

    # not needed in the baseline model:
    # def eval(self, corpus: Corpus, predicted: list) -> Result: ...
